"""OpenAI Chat Completions wire dialect."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

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

MaxTokensField = Literal["max_tokens", "max_completion_tokens"]


def serialize_tool_schemas_openai(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """Encode canonical tool specs as OpenAI function schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": json.loads(json.dumps(t.parameters, sort_keys=True)),
            },
        }
        for t in canonical_tool_specs(tools)
    ]


@dataclass(frozen=True)
class OpenAIChatCompat:
    """Minimal per-endpoint compatibility switches observed in the wild."""

    max_tokens_field: MaxTokensField | None = None
    include_tool_result_name: bool = False
    non_null_assistant_content: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> OpenAIChatCompat:
        raw_field = mapping.get("max_tokens_field")
        tokens_field: MaxTokensField | None = None
        if raw_field in ("max_tokens", "max_completion_tokens"):
            tokens_field = raw_field
        raw_extra = mapping.get("extra_body")
        return cls(
            max_tokens_field=tokens_field,
            include_tool_result_name=bool(mapping.get("include_tool_result_name", False)),
            non_null_assistant_content=bool(mapping.get("non_null_assistant_content", False)),
            extra_body=dict(raw_extra) if isinstance(raw_extra, Mapping) else {},
        )


@dataclass(frozen=True)
class OpenAIChatDialect:
    """Posts ``/chat/completions`` and parses the completion response."""

    compat: OpenAIChatCompat = field(default_factory=OpenAIChatCompat)
    max_tokens: int | None = None

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
        headers: dict[str, str] = {"Content-Type": "application/json"}
        headers.update(endpoint.headers)
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"

        url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status_code >= 400:
            raise HttpStatusError(resp.status_code, url, resp.text[:500])

        data = resp.json()
        choice = data["choices"][0]["message"]
        parsed_calls: list[ToolCall] = []
        for raw_tc in choice.get("tool_calls") or []:
            fn = raw_tc["function"]
            raw_args = fn.get("arguments") or "{}"
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            parsed_calls.append(ToolCall(id=raw_tc["id"], name=fn["name"], arguments=args))

        usage = data.get("usage") or {}
        return LLMResponse(
            content=choice.get("content") or "",
            tool_calls=tuple(parsed_calls),
            usage=UsageStats(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
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
        payload_messages: list[dict[str, Any]] = []
        for msg in messages:
            content: str | None = msg.content
            if content == "" and msg.tool_calls and not self.compat.non_null_assistant_content:
                content = None
            item: dict[str, Any] = {"role": msg.role, "content": content}
            if msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, sort_keys=True),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id is not None:
                item["tool_call_id"] = msg.tool_call_id
            if self.compat.include_tool_result_name and msg.name is not None:
                item["name"] = msg.name
            payload_messages.append(item)

        body: dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = serialize_tool_schemas_openai(tools)
        if self.max_tokens is not None and self.compat.max_tokens_field is not None:
            body[self.compat.max_tokens_field] = self.max_tokens
        for key, value in self.compat.extra_body.items():
            body.setdefault(key, value)
        return body
