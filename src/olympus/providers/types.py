"""Vendor-neutral provider contracts shared by every dialect and caller."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

_MAX_BODY_PREVIEW = 500


class ProviderError(RuntimeError):
    """Raised when an upstream LLM request cannot be completed."""


class HttpStatusError(ProviderError):
    """HTTP-level failure carrying non-secret diagnostic context."""

    def __init__(self, status_code: int, url: str, body_preview: str) -> None:
        self.status_code = status_code
        self.url = url
        self.body_preview = body_preview[:_MAX_BODY_PREVIEW]
        super().__init__(f"HTTP {status_code} from {url}: {self.body_preview}")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: UsageStats = field(default_factory=UsageStats)
    raw_model: str = ""


class LLMProvider(Protocol):
    model_name: str

    def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
    ) -> LLMResponse: ...


ScriptedResponder = Callable[[Sequence[ChatMessage], Sequence[ToolSpec]], LLMResponse]


class MockLLMProvider:
    """Deterministic offline provider for simulation runs, unit tests, and replay."""

    def __init__(
        self,
        responses: Sequence[LLMResponse] | None = None,
        responder: ScriptedResponder | None = None,
        model_name: str = "mock-olympus-v1",
    ) -> None:
        self.model_name = model_name
        self._queue = list(responses or [])
        self._responder = responder
        self.recorded_requests: list[tuple[tuple[ChatMessage, ...], tuple[ToolSpec, ...]]] = []
        self.recorded_options: list[dict[str, Any]] = []

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
        del temperature
        self.recorded_requests.append((tuple(messages), tuple(tools)))
        self.recorded_options.append(
            {
                "enable_thinking": enable_thinking,
                "stream": stream,
                "timeout": timeout,
            }
        )
        if self._responder is not None:
            return self._responder(messages, tools)
        if not self._queue:
            raise RuntimeError("MockLLMProvider exhausted queued responses")
        return self._queue.pop(0)


def canonical_tool_specs(tools: Sequence[ToolSpec]) -> list[ToolSpec]:
    """Sort tools deterministically by name so prompt prefixes remain KV-cache stable."""
    return sorted(tools, key=lambda t: t.name)
