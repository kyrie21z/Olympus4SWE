"""Anthropic Messages wire dialect."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import anthropic
import httpx
from anthropic.types import MessageParam, TextBlock, ToolUnionParam, ToolUseBlock

from olympus.providers.auth import EndpointTarget
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
        timeout: float = 180.0,
    ) -> LLMResponse:
        conversation, system_text = self._serialize_messages(messages)
        url = f"{endpoint.base_url.rstrip('/')}/v1/messages"
        client = anthropic.Anthropic(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            default_headers=dict(endpoint.headers),
            timeout=timeout,
            max_retries=0,
            http_client=self.http_client,
        )
        request_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": cast(Sequence[MessageParam], conversation),
            "temperature": temperature,
        }
        if system_text:
            request_kwargs["system"] = system_text
        if tools:
            request_kwargs["tools"] = cast(
                Sequence[ToolUnionParam],
                self._serialize_tools(tools),
            )
        if not endpoint.api_key and not any(
            key.lower() == "x-api-key" for key in endpoint.headers
        ):
            request_kwargs["extra_headers"] = {"X-Api-Key": anthropic.Omit()}

        try:
            message = client.messages.create(**request_kwargs)
        except anthropic.APIStatusError as exc:
            raise HttpStatusError(exc.status_code, url, exc.response.text[:500]) from exc
        finally:
            if self.http_client is None:
                client.close()

        texts: list[str] = []
        parsed_calls: list[ToolCall] = []
        for block in message.content or []:
            if isinstance(block, TextBlock):
                texts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                parsed_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=cast(dict[str, Any], block.input),
                    )
                )

        usage = message.usage
        return LLMResponse(
            content="\n".join(texts),
            tool_calls=tuple(parsed_calls),
            usage=UsageStats(
                input_tokens=usage.input_tokens if usage is not None else 0,
                output_tokens=usage.output_tokens if usage is not None else 0,
            ),
            raw_model=message.model or model,
        )

    def _serialize_messages(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[list[dict[str, Any]], str]:
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
        return conversation, "\n\n".join(system_parts)

    def _serialize_tools(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": json.loads(json.dumps(t.parameters, sort_keys=True)),
            }
            for t in canonical_tool_specs(tools)
        ]
