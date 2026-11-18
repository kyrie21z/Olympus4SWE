"""Anthropic Messages wire dialect."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

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

_ANTHROPIC_VERSION = "2023-06-01"


@dataclass(frozen=True)
class AnthropicMessagesDialect:
    """Posts ``/v1/messages`` and parses content blocks."""

    max_tokens: int = 4096

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
        body = self._build_body(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
        )
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        if endpoint.api_key:
            headers["x-api-key"] = endpoint.api_key
        headers.update(endpoint.headers)

        url = f"{endpoint.base_url.rstrip('/')}/v1/messages"
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status_code >= 400:
            raise HttpStatusError(resp.status_code, url, resp.text[:500])

        data = resp.json()
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
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            ),
            raw_model=str(data.get("model") or model),
        )

    def _build_body(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
        temperature: float,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        convo: list[dict[str, Any]] = []
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
                convo.append({"role": "assistant", "content": blocks or msg.content})
            elif msg.role == "tool":
                convo.append(
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
                convo.append({"role": "user", "content": msg.content})

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": json.loads(json.dumps(t.parameters, sort_keys=True)),
                }
                for t in canonical_tool_specs(tools)
            ]
        return body
