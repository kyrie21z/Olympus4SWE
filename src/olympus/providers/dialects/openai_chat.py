"""OpenAI Chat Completions wire dialect."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import httpx
import openai
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)

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

# The SDK refuses to construct without credentials; keyless endpoints strip the
# synthesized Authorization header on each request instead.
_KEYLESS_PLACEHOLDER = "unused"


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
    """Sends ``/chat/completions`` through the official OpenAI SDK."""

    compat: OpenAIChatCompat = field(default_factory=OpenAIChatCompat)
    max_tokens: int | None = None
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
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": cast(
                Sequence[ChatCompletionMessageParam],
                self._serialize_messages(messages),
            ),
            "temperature": temperature,
        }
        if tools:
            request_kwargs["tools"] = cast(
                Sequence[ChatCompletionToolUnionParam],
                serialize_tool_schemas_openai(tools),
            )
        if self.max_tokens is not None and self.compat.max_tokens_field is not None:
            request_kwargs[self.compat.max_tokens_field] = self.max_tokens
        passthrough = {
            key: value
            for key, value in self.compat.extra_body.items()
            if key not in request_kwargs
        }
        if passthrough:
            request_kwargs["extra_body"] = passthrough

        explicit_auth: str | None = None
        default_headers: dict[str, str] = {}
        has_org = False
        has_project = False
        for key, value in endpoint.headers.items():
            lowered = key.lower()
            if lowered == "authorization":
                explicit_auth = value
                continue
            if lowered == "openai-organization":
                has_org = True
            elif lowered == "openai-project":
                has_project = True
            default_headers[key] = value

        extra_headers: dict[str, Any] = {}
        if not has_org:
            extra_headers["OpenAI-Organization"] = openai.Omit()
        if not has_project:
            extra_headers["OpenAI-Project"] = openai.Omit()
        if not endpoint.api_key:
            extra_headers["Authorization"] = (
                explicit_auth if explicit_auth is not None else openai.Omit()
            )
        if extra_headers:
            request_kwargs["extra_headers"] = extra_headers

        owned_client = (
            httpx.Client(timeout=timeout, follow_redirects=False)
            if self.http_client is None
            else None
        )
        client = openai.OpenAI(
            api_key=endpoint.api_key or _KEYLESS_PLACEHOLDER,
            base_url=endpoint.base_url,
            default_headers=default_headers,
            timeout=timeout,
            max_retries=0,
            http_client=self.http_client if self.http_client is not None else owned_client,
        )

        try:
            raw = client.chat.completions.with_raw_response.create(**request_kwargs)
        except openai.APIStatusError as exc:
            raise HttpStatusError(
                exc.status_code, str(exc.request.url), exc.response.text[:500]
            ) from exc
        except openai.APIConnectionError as exc:
            if isinstance(exc.__cause__, httpx.HTTPError):
                raise exc.__cause__ from exc
            raise
        finally:
            if owned_client is not None:
                owned_client.close()

        data = raw.http_response.json()
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
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            ),
            raw_model=str(data.get("model") or model),
        )

    def _serialize_messages(self, messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
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
        return payload_messages
