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
        timeout: float = 180.0,
    ) -> LLMResponse: ...


__all__ = [
    "AnthropicMessagesDialect",
    "Dialect",
    "OpenAIChatCompat",
    "OpenAIChatDialect",
    "serialize_tool_schemas_openai",
]
