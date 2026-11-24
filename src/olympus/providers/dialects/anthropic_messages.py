"""Anthropic Messages wire dialect."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import anthropic
import httpx
from anthropic.types import MessageParam, ToolUnionParam

from olympus.providers.auth import EndpointTarget
from olympus.providers.dialects.http import (
    DEFAULT_READ_TIMEOUT_SECONDS,
    build_keepalive_httpx_client,
)
from olympus.providers.types import (
    ChatMessage,
    HttpStatusError,
    LLMResponse,
    ToolCall,
    ToolSpec,
    UsageStats,
    canonical_tool_specs,
)


@dataclass(frozen=True)
class AnthropicMessagesDialect:
    """Sends ``/v1/messages`` through the official Anthropic SDK."""

    wire_api: str = "anthropic-messages"
    max_tokens: int = 4096
    http_client: httpx.Client | None = None

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
    ) -> LLMResponse:
        del enable_thinking, stream
        conversation, system_parts = self._serialize_messages(messages)
        request_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": cast(Sequence[MessageParam], conversation),
            "temperature": temperature,
        }
        if system_parts:
            request_kwargs["system"] = "\n\n".join(system_parts)
        if tools:
            request_kwargs["tools"] = cast(
                Sequence[ToolUnionParam],
                self._serialize_tools(tools),
            )

        explicit_key: str | None = None
        has_auth = False
        default_headers: dict[str, str] = {}
        for key, value in endpoint.headers.items():
            lowered = key.lower()
            if lowered == "x-api-key":
                explicit_key = value
                continue
            if lowered == "authorization":
                has_auth = True
            default_headers[key] = value

        effective_key = explicit_key if explicit_key is not None else endpoint.api_key
        extra_headers: dict[str, Any] = {}
        if not effective_key:
            extra_headers["X-Api-Key"] = anthropic.Omit()
        if not has_auth:
            extra_headers["Authorization"] = anthropic.Omit()
        if extra_headers:
            request_kwargs["extra_headers"] = extra_headers

        owned_client = (
            build_keepalive_httpx_client(timeout=timeout, follow_redirects=False)
            if self.http_client is None
            else None
        )
        client = anthropic.Anthropic(
            api_key=effective_key,
            base_url=endpoint.base_url,
            default_headers=default_headers,
            timeout=timeout,
            max_retries=0,
            http_client=self.http_client if self.http_client is not None else owned_client,
        )

        try:
            raw = client.messages.with_raw_response.create(**request_kwargs)
        except anthropic.APIStatusError as exc:
            raise HttpStatusError(
                exc.status_code, str(exc.request.url), exc.response.text[:500]
            ) from exc
        except anthropic.APIConnectionError as exc:
            if isinstance(exc.__cause__, httpx.HTTPError):
                # Break the SDK's exc.__cause__ link so re-raising cannot form a
                # circular __cause__ chain (which hangs traceback formatters).
                raise exc.__cause__ from None
            raise
        finally:
            if owned_client is not None:
                owned_client.close()

        data = raw.http_response.json()
        texts: list[str] = []
        parsed_calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                parsed_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input") or {},
                    )
                )

        usage = data.get("usage") or {}
        return LLMResponse(
            content="\n".join(texts),
            tool_calls=tuple(parsed_calls),
            usage=UsageStats(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            ),
            raw_model=str(data.get("model") or model),
        )

    def _serialize_messages(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        system_parts: list[str] = []
        conversation: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                conversation.append({"role": "assistant", "content": blocks or msg.content})
            elif msg.role == "tool":
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content,
                            }
                        ],
                    }
                )
            else:
                conversation.append({"role": "user", "content": msg.content})
        return conversation, system_parts

    def _serialize_tools(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": json.loads(json.dumps(t.parameters, sort_keys=True)),
            }
            for t in canonical_tool_specs(tools)
        ]
