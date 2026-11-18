"""Shared helpers for structured LLM output parsing and role prompts."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

from pydantic import BaseModel, ValidationError

from olympus.harness.loop import AgentLoop
from olympus.providers.base import ChatMessage, LLMProvider
from olympus.tools.aci import WorkspaceToolkit

_OUTER_FENCE_RE = re.compile(r"^\s*```(?:json|[a-zA-Z0-9_-]+)?\s*\n(.*)\n```\s*$", re.DOTALL)
_FENCE_RE = re.compile(r"```([a-zA-Z0-9_-]*)\s*\n?(.*?)\n?```", re.DOTALL)
_LINE_OBJECT_RE = re.compile(r"(?m)^(?:\{|\[)")
_SUBMIT_ARG_KEYS = (
    "plan",
    "spec",
    "design",
    "verdict",
    "result",
    "output",
    "payload",
    "data",
    "json",
)
_SCHEMA_META_KEYS = frozenset(
    {
        "$schema",
        "$comment",
        "$id",
        "$defs",
        "$ref",
        "$anchor",
        "$vocabulary",
        "$dynamicRef",
        "$dynamicAnchor",
    }
)


def format_workspace_inventory(toolkit: WorkspaceToolkit, *, max_files: int = 40) -> str:
    """Format a bounded deterministic inventory of existing workspace paths."""
    files, total = toolkit.list_relative_files(max_files=max_files)
    if not files:
        return "Observed workspace files: (none)"
    suffix = f" (+{total - len(files)} more)" if total > len(files) else ""
    return f"Observed workspace files ({len(files)}/{total}): " + ", ".join(files) + suffix


def strip_schema_meta_keys(value: Any) -> Any:
    """Recursively drop JSON Schema metadata keys while keeping extra='forbid' strictness."""
    if isinstance(value, dict):
        return {
            key: strip_schema_meta_keys(val)
            for key, val in value.items()
            if not (isinstance(key, str) and (key in _SCHEMA_META_KEYS or key.startswith("$")))
        }
    if isinstance(value, list):
        return [strip_schema_meta_keys(item) for item in value]
    return value


def _is_valid_json(candidate: str) -> bool:
    if not candidate:
        return False
    with suppress(json.JSONDecodeError):
        json.loads(candidate)
        return True
    return False


def _largest_decodable_json(text: str) -> str | None:
    """Return the longest top-level JSON object/array span cleanly decodable from text."""
    decoder = json.JSONDecoder()
    best: str | None = None
    idx = 0
    length = len(text)
    while idx < length:
        if text[idx] in ("{", "["):
            try:
                obj, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                idx += 1
                continue
            if isinstance(obj, (dict, list)):
                candidate = text[idx:end]
                if best is None or len(candidate) >= len(best):
                    best = candidate
            idx = end
            continue
        idx += 1
    return best


def extract_json_payload(raw_text: str) -> str:
    """Extract outer JSON without letting inline prose dicts or inner code fences hijack it."""
    stripped = raw_text.strip()
    if not stripped:
        return ""
    if _is_valid_json(stripped):
        return stripped
    outer_match = _OUTER_FENCE_RE.match(stripped)
    if outer_match:
        outer_body = outer_match.group(1).strip()
        if _is_valid_json(outer_body):
            return outer_body
    largest = _largest_decodable_json(stripped)
    if largest is not None:
        return largest
    fences = [(m.group(1).lower(), m.group(2).strip()) for m in _FENCE_RE.finditer(stripped)]
    for tag, body in fences:
        if tag in ("", "json") and _is_valid_json(body):
            return body
    if outer_match:
        return outer_match.group(1).strip()
    if not stripped.startswith(("{", "[")):
        line_start = _LINE_OBJECT_RE.search(stripped)
        if line_start is not None:
            return stripped[line_start.start() :]
        for tag, body in fences:
            if tag in ("", "json"):
                return body
    start_obj = stripped.find("{")
    end_obj = stripped.rfind("}")
    if start_obj != -1 and end_obj > start_obj:
        return stripped[start_obj : end_obj + 1]
    return stripped


def parse_model_json[T: BaseModel](raw_text: str, model_cls: type[T]) -> T:
    """Extract JSON, strip schema meta-keys ($comment/$schema), and validate strictly."""
    candidate = extract_json_payload(raw_text)
    data = strip_schema_meta_keys(json.loads(candidate))
    return model_cls.model_validate(data)


def _validate_candidate_value[T: BaseModel](raw: Any, model_cls: type[T]) -> T | None:
    if isinstance(raw, str):
        with suppress(json.JSONDecodeError, ValidationError, ValueError):
            return parse_model_json(raw, model_cls)
    elif isinstance(raw, dict):
        with suppress(ValidationError, ValueError):
            return model_cls.model_validate(strip_schema_meta_keys(raw))
    return None


def _salvage_from_messages[T: BaseModel](
    messages: Sequence[ChatMessage],
    model_cls: type[T],
) -> T | None:
    """Recover a valid model payload from the last assistant turn or hallucinated submit tools."""
    last_assistant = next((m for m in reversed(messages) if m.role == "assistant"), None)
    if last_assistant is None:
        return None
    if last_assistant.content.strip():
        parsed = _validate_candidate_value(last_assistant.content, model_cls)
        if parsed is not None:
            return parsed
    for tc in reversed(last_assistant.tool_calls):
        args = tc.arguments
        for key in _SUBMIT_ARG_KEYS:
            if key in args:
                parsed = _validate_candidate_value(args[key], model_cls)
                if parsed is not None:
                    return parsed
        if len(args) == 1:
            only_val = next(iter(args.values()))
            parsed = _validate_candidate_value(only_val, model_cls)
            if parsed is not None:
                return parsed
        parsed = _validate_candidate_value(args, model_cls)
        if parsed is not None:
            return parsed
    return None


def run_structured_loop[T: BaseModel](
    *,
    loop: AgentLoop,
    system_prompt: str,
    user_prompt: str,
    model_cls: type[T],
    max_parse_retries: int = 2,
) -> T:
    """Run an AgentLoop and validate final JSON response against a Pydantic model."""
    last_err = ""
    prev_synth = loop.synthesize_on_max_steps
    loop.synthesize_on_max_steps = True
    try:
        res = loop.run(system_prompt=system_prompt, user_prompt=user_prompt)
        for attempt in range(max_parse_retries + 1):
            if not res.completed:
                if res.stop_reason.startswith(("budget_exhausted", "provider_error")):
                    raise RuntimeError(
                        f"AgentLoop stopped before producing final output: {res.stop_reason}"
                    )
                salvaged = _salvage_from_messages(res.messages, model_cls)
                if salvaged is not None:
                    return salvaged
                last_err = f"stopped early ({res.stop_reason})"
                if attempt < max_parse_retries:
                    schema_hint = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
                    correction = (
                        f"[STEP LIMIT REACHED: {res.stop_reason}]\n"
                        "Stop calling tools immediately. Output ONLY a single raw JSON object "
                        "starting with '{' matching schema "
                        f"(no prose, no $comment/$schema/extra keys):\n{schema_hint}"
                    )
                    res = loop.complete_repair(
                        prior_messages=res.messages,
                        correction=correction,
                        step=attempt + 1,
                    )
                    continue
                raise RuntimeError(
                    f"AgentLoop stopped before producing final output: {res.stop_reason}"
                )
            try:
                return parse_model_json(res.final_content, model_cls)
            except (json.JSONDecodeError, ValidationError) as exc:
                salvaged = _salvage_from_messages(res.messages, model_cls)
                if salvaged is not None:
                    return salvaged
                last_err = str(exc)
                if attempt < max_parse_retries:
                    schema_hint = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
                    empty_note = (
                        "Previous output was empty. "
                        if not res.final_content.strip()
                        else f"Error: {last_err}\n"
                    )
                    correction = (
                        "[PREVIOUS OUTPUT FAILED VALIDATION]\n"
                        f"{empty_note}"
                        "Do NOT call any tools and do NOT include leading prose. Return ONLY "
                        "a single valid JSON object starting with '{' matching schema "
                        f"(no $comment, $schema, or extra keys):\n{schema_hint}"
                    )
                    res = loop.complete_repair(
                        prior_messages=res.messages,
                        correction=correction,
                        step=attempt + 1,
                    )
        raise ValueError(f"Failed to parse {model_cls.__name__} JSON: {last_err}")
    finally:
        loop.synthesize_on_max_steps = prev_synth


def build_read_only_loop(
    *,
    provider: LLMProvider,
    loop_kwargs: dict[str, object],
) -> AgentLoop:
    """Convenience constructor helper preserving strict keyword args."""
    return AgentLoop(provider=provider, **loop_kwargs)  # type: ignore[arg-type]
