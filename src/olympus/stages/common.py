"""Shared helpers for structured LLM output parsing and role prompts."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from olympus.harness.loop import AgentLoop
from olympus.providers.base import LLMProvider
from olympus.tools.aci import WorkspaceToolkit

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
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


def extract_json_payload(raw_text: str) -> str:
    """Extract JSON object/array from plain text or fenced Markdown blocks."""
    stripped = raw_text.strip()
    match = _FENCE_RE.search(stripped)
    if match:
        return match.group(1).strip()
    # Fallback: locate outermost braces
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


def run_structured_loop[T: BaseModel](
    *,
    loop: AgentLoop,
    system_prompt: str,
    user_prompt: str,
    model_cls: type[T],
    max_parse_retries: int = 1,
) -> T:
    """Run an AgentLoop and validate final JSON response against a Pydantic model."""
    last_err = ""
    prev_synth = loop.synthesize_on_max_steps
    loop.synthesize_on_max_steps = True
    try:
        res = loop.run(system_prompt=system_prompt, user_prompt=user_prompt)
        for attempt in range(max_parse_retries + 1):
            if not res.completed:
                raise RuntimeError(
                    f"AgentLoop stopped before producing final output: {res.stop_reason}"
                )
            try:
                return parse_model_json(res.final_content, model_cls)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_err = str(exc)
                if attempt < max_parse_retries:
                    schema_hint = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
                    correction = (
                        f"[PREVIOUS OUTPUT FAILED VALIDATION]\n"
                        f"Error: {last_err}\n"
                        "Fix the validation errors using what you already observed. "
                        "Return ONLY valid JSON matching schema (no $comment, $schema, "
                        f"or extra keys):\n{schema_hint}"
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
