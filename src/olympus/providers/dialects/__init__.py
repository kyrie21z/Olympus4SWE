"""Wire dialects: serialization and HTTP transport only, no vendor identity.

A dialect knows one protocol shape (OpenAI Chat Completions or Anthropic
Messages). Provider identity, credentials and failover live in
``olympus.providers.auth`` / ``olympus.providers.registry``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from olympus.providers.auth import EndpointTarget
from olympus.providers.dialects.anthropic_messages import AnthropicMessagesDialect
from olympus.providers.dialects.http import (
    DEFAULT_READ_TIMEOUT_SECONDS,
    build_keepalive_httpx_client,
    tcp_keepalive_socket_options,
)
from olympus.providers.dialects.openai_chat import (
    OpenAIChatCompat,
    OpenAIChatDialect,
    serialize_tool_schemas_openai,
)
from olympus.providers.types import ChatMessage, LLMResponse, ToolSpec


class Dialect(Protocol):
    def send(
        self,
        endpoint: EndpointTarget,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
        timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
        enable_thinking: bool | None = None,
        stream: bool = False,
    ) -> LLMResponse: ...


__all__ = [
    "AnthropicMessagesDialect",
    "DEFAULT_READ_TIMEOUT_SECONDS",
    "Dialect",
    "OpenAIChatCompat",
    "OpenAIChatDialect",
    "build_keepalive_httpx_client",
    "serialize_tool_schemas_openai",
    "tcp_keepalive_socket_options",
]
