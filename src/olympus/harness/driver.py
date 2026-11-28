"""Execution contract for built-in and external agent harnesses."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Protocol

from pydantic import Field, StringConstraints, field_validator

from olympus.core.contracts import StrictModel
from olympus.providers.base import LLMProvider, MockLLMProvider

if TYPE_CHECKING:
    from olympus.core.store import RunStore

DriverEvent = Callable[[str, dict[str, Any]], None]
DriverFactory = Callable[[], "AgentDriver"]
NonEmptySetting = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Pi ``--thinking`` levels per its documented thinkingLevelMap scale.
VALID_THINKING_LEVELS = frozenset(
    {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
)
_VALID_THINKING_LEVELS = VALID_THINKING_LEVELS


class DriverTurnRequest(StrictModel):
    workspace_path: Path
    env_root: Path
    system_prompt: str
    user_prompt: str
    actor: str
    task_id: str | None = None
    lease_generation: int | None = None
    read_only: bool = False
    disable_tools: bool = False
    strict_scope: bool = True
    write_scope: list[str] = Field(default_factory=list)
    forbidden_globs: list[str] = Field(default_factory=list)
    max_steps: int | None = Field(default=None, ge=1)
    trace_path: Path | None = None
    model: str | None = None
    pi_provider: str | None = None
    thinking_level: str | None = None
    context_window: int | None = Field(default=None, ge=1024)


@dataclass(frozen=True)
class TurnOutcome:
    completed: bool
    stop_reason: str
    final_content: str
    steps_taken: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_reasoning_tokens: int = 0
    receipt: dict[str, Any] = field(default_factory=dict)
    driver_name: str = ""


class ActiveTurnHandle(Protocol):
    @property
    def supports_steer(self) -> bool: ...

    def steer(self, guidance: str, *, source: str) -> bool: ...

    def cancel(self, *, reason: str) -> None: ...


class AgentDriver(Protocol):
    """Publish a handle before execution and emit callbacks during the turn.

    Tool milestones use agent.tool_started / agent.tool_finished, with step,
    tool and (for completion) status, duration_ms and error_signature.
    agent.steered carries the source of guidance. text_delta carries text,
    with replace=True when a driver supplies a complete message.
    Drivers own local JSONL traces. Finished tools use event="tool_end" with
    the milestone's tool fields plus arguments and output. Text records use
    event="text_delta" with the corresponding text payload.
    """

    name: str

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: DriverEvent | None = None,
        on_handle: Callable[[ActiveTurnHandle], None] | None = None,
    ) -> TurnOutcome: ...


_STAGE_ROLES: frozenset[str] = frozenset(
    {"stage", "explorer", "architect", "planner", "aligner", "designer"}
)
_REVIEWER_ROLES: frozenset[str] = frozenset(
    {
        "reviewer",
        "interrogator",
        "thermo_reviewer",
        "verifier",
        "judge",
        "arena_judge",
        "critic",
    }
)
_WORKER_ROLES: frozenset[str] = frozenset(
    {"worker", "builder", "hygiene", "arena_runner", "hillclimber"}
)


def _omit_empty_mapping(value: object) -> bool:
    return not value


class ExecutionDriversConfig(StrictModel):
    stage_driver: NonEmptySetting = "auto"
    worker_driver: NonEmptySetting = "auto"
    reviewer_driver: NonEmptySetting = "auto"
    role_drivers: dict[NonEmptySetting, NonEmptySetting] = Field(
        default_factory=dict,
        exclude_if=_omit_empty_mapping,
    )
    monitor_model: NonEmptySetting | None = None
    pi_provider: NonEmptySetting | None = None
    thinking_level: NonEmptySetting | None = None
    context_window: int | None = Field(default=None, ge=1024)

    @field_validator("role_drivers", mode="before")
    @classmethod
    def _normalize_role_drivers(cls, value: object) -> object:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(
                f"role_drivers must be a mapping, got {type(value).__name__}"
            )
        normalized: dict[str, str] = {}
        for raw_role, raw_spec in value.items():
            if not isinstance(raw_role, str) or not raw_role.strip():
                raise ValueError(f"role_drivers keys must be non-empty strings, got {raw_role!r}")
            if not isinstance(raw_spec, str) or not raw_spec.strip():
                raise ValueError(
                    f"role_drivers[{raw_role!r}] must be a non-empty string, got {raw_spec!r}"
                )
            normalized[raw_role.strip().lower()] = raw_spec.strip()
        return normalized

    @field_validator("thinking_level", mode="before")
    @classmethod
    def _normalize_thinking_level(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"thinking_level must be a string, got {type(value).__name__}")
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized not in VALID_THINKING_LEVELS:
            raise ValueError(
                f"Invalid thinking_level {value!r}; expected one of "
                f"{', '.join(sorted(VALID_THINKING_LEVELS))}"
            )
        return normalized

    def resolve_driver_for_role(self, role: str | None) -> str:
        """Return the driver spec for ``role``, falling back cleanly to coarse slots."""
        if role is not None:
            normalized = role.strip().lower()
            if normalized:
                explicit = self.role_drivers.get(normalized)
                if explicit is not None:
                    return explicit
                if normalized in _REVIEWER_ROLES:
                    return self.reviewer_driver
                if normalized in _STAGE_ROLES:
                    return self.stage_driver
                if normalized in _WORKER_ROLES:
                    return self.worker_driver
        return self.worker_driver

    @classmethod
    def resolve(
        cls,
        *,
        stage_driver: str | None = None,
        worker_driver: str | None = None,
        reviewer_driver: str | None = None,
        role_drivers: Mapping[str, str] | None = None,
        monitor_model: str | None = None,
        pi_provider: str | None = None,
        thinking_level: str | None = None,
        context_window: int | None = None,
    ) -> ExecutionDriversConfig:
        explicit = {
            "stage_driver": stage_driver,
            "worker_driver": worker_driver,
            "reviewer_driver": reviewer_driver,
            "monitor_model": monitor_model,
            "pi_provider": pi_provider,
            "thinking_level": thinking_level,
        }
        values: dict[str, Any] = {}
        for field_name, value in explicit.items():
            if value is not None:
                values[field_name] = value
            elif setting := os.getenv(f"OLYMPUS_{field_name.upper()}", "").strip():
                values[field_name] = setting
        merged_roles: dict[str, str] = {}
        raw_roles_env = os.getenv("OLYMPUS_ROLE_DRIVERS", "").strip()
        if raw_roles_env:
            try:
                parsed_roles = json.loads(raw_roles_env)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid OLYMPUS_ROLE_DRIVERS JSON: {exc.msg}"
                ) from exc
            if not isinstance(parsed_roles, Mapping):
                raise ValueError("Invalid OLYMPUS_ROLE_DRIVERS: expected a JSON object")
            for raw_role, raw_spec in parsed_roles.items():
                key = raw_role.strip().lower() if isinstance(raw_role, str) else raw_role
                merged_roles[key] = raw_spec
        if role_drivers is not None:
            for raw_role, raw_spec in role_drivers.items():
                key = raw_role.strip().lower() if isinstance(raw_role, str) else raw_role
                merged_roles[key] = raw_spec
        if merged_roles:
            values["role_drivers"] = merged_roles
        raw_window = os.getenv("OLYMPUS_CONTEXT_WINDOW", "").strip()
        if context_window is not None:
            values["context_window"] = context_window
        elif raw_window:
            try:
                values["context_window"] = int(raw_window)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid OLYMPUS_CONTEXT_WINDOW {raw_window!r}: not an integer"
                ) from exc
        return cls.model_validate(values)


def _is_external_driver_eligible(provider: LLMProvider | None) -> bool:
    """Allow ``auto`` to choose Pi only for real remote providers (never test doubles)."""
    if provider is None or isinstance(provider, MockLLMProvider):
        return False
    if str(getattr(provider, "provider_name", "") or "").strip().lower() == "mock":
        return False
    if getattr(provider, "supports_external_driver", False) is True:
        return True
    from olympus.providers.auth import PooledProvider

    return isinstance(provider, PooledProvider)


def resolve_driver_spec(spec: str, *, provider: LLMProvider | None) -> str:
    """Resolve ``auto``/``auto@<model>`` against the provider kind and Pi availability.

    ``auto`` becomes ``pi`` (carrying the explicit or provider model) only on a
    real remote provider when the cached Pi probe succeeds. Mock/in-process test
    providers and missing binaries fall back to ``native`` so offline tests and
    zero-Node environments keep working. Explicit specs pass through untouched,
    preserving strict ``pi`` failure and ``native`` routing semantics.
    """
    name, _, raw_model = spec.partition("@")
    if name != "auto":
        return spec
    if not _is_external_driver_eligible(provider):
        return "native"
    from olympus.harness.pi import pi_available

    if not pi_available():
        return "native"
    model = raw_model.strip() or str(getattr(provider, "model_name", "") or "").strip()
    return f"pi@{model}" if model else "pi"


class DriverRegistry:
    """Resolve driver specs against explicit drivers, lazy factories, and the native factory."""

    def __init__(
        self,
        drivers: Mapping[str, AgentDriver] | None = None,
        *,
        factories: Mapping[str, DriverFactory] | None = None,
    ) -> None:
        self._drivers = dict(drivers or {})
        self._factories = dict(factories or {})
        self._built: dict[str, AgentDriver] = {}

    def with_factories(self, factories: Mapping[str, DriverFactory]) -> DriverRegistry:
        """Return a registry that keeps explicit drivers and adds lazy default drivers."""
        merged = dict(self._factories)
        merged.update(factories)
        return DriverRegistry(self._drivers, factories=merged)

    @staticmethod
    def _split_spec(spec: str) -> tuple[str, str | None]:
        name, separator, raw_model = spec.partition("@")
        model = raw_model.strip() or None
        if separator and model is None:
            raise ValueError(f"Missing model override in driver spec {spec!r}")
        return name, model

    def _build(self, name: str) -> AgentDriver:
        if name in self._drivers:
            return self._drivers[name]
        if name in self._factories:
            built = self._built.get(name)
            if built is None:
                built = self._factories[name]()
                self._built[name] = built
            return built
        registered = sorted({"native", *self._drivers, *self._factories})
        raise ValueError(
            f"Unknown agent driver {name!r}; registered drivers: "
            f"{', '.join(registered)}"
        )

    def resolve(
        self, spec: str, *, native_factory: Callable[[str | None], AgentDriver]
    ) -> tuple[AgentDriver, str | None]:
        name, model = self._split_spec(spec)
        driver = native_factory(model) if name == "native" else self._build(name)
        return driver, model

    def resolve_external(self, spec: str) -> tuple[AgentDriver, str | None]:
        """Resolve a spec that must not use the built-in native driver."""
        name, model = self._split_spec(spec)
        if name == "native":
            raise ValueError("native is not an external agent driver")
        return self._build(name), model


def default_driver_factories(
    *,
    store: RunStore | None,
    run_id: str | None,
    provider: LLMProvider | None = None,
    execution_drivers: ExecutionDriversConfig | None = None,
) -> dict[str, DriverFactory]:
    """Lazily build the external drivers shipped with Olympus (never at import time)."""
    from olympus.harness.pi import PiDriver

    settings = execution_drivers or ExecutionDriversConfig()
    pi_provider = settings.pi_provider
    thinking_level = settings.thinking_level
    context_window = settings.context_window
    return {
        "pi": lambda: PiDriver(
            store=store,
            run_id=run_id,
            provider=provider,
            pi_provider=pi_provider,
            thinking_level=thinking_level,
            context_window=context_window,
        )
    }
