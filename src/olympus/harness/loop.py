"""Transparent Append-Only AgentLoop with KV-prefix stability, budget guard, and trace log."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olympus.core.contracts import new_id, utc_now
from olympus.core.fingerprint import content_hash
from olympus.core.store import RunStore
from olympus.harness.hooks import PreToolUseHook
from olympus.harness.observer import Observer
from olympus.harness.receipt import build_turn_receipt
from olympus.harness.trim import shrink_seen
from olympus.providers.base import ChatMessage, LLMProvider, ToolSpec, canonical_tool_specs
from olympus.tools.aci import WorkspaceToolkit


@dataclass(frozen=True)
class LoopStepRecord:
    step: int
    prefix_hash: str
    message_count: int
    assistant_content: str
    tool_calls: list[dict[str, Any]]
    tool_outputs: list[dict[str, str]]
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LoopResult:
    completed: bool
    stop_reason: str
    final_content: str
    steps_taken: int
    messages: tuple[ChatMessage, ...] = field(default_factory=tuple)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    receipt: dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    """Append-Only execution loop maintaining static prefix and full error preservation."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        toolkit: WorkspaceToolkit | None = None,
        store: RunStore | None = None,
        run_id: str | None = None,
        actor: str = "agent",
        max_steps: int = 20,
        trace_path: Path | None = None,
        synthesize_on_max_steps: bool = False,
        observer: Observer | None = None,
        pre_tool_hook: PreToolUseHook | None = None,
        trim_seen: bool = True,
        task_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.toolkit = toolkit
        self.store = store
        self.run_id = run_id
        self.actor = actor
        self.max_steps = max_steps
        self.trace_path = trace_path
        self.synthesize_on_max_steps = synthesize_on_max_steps
        self.observer = observer
        self.pre_tool_hook = pre_tool_hook
        self.trim_seen = trim_seen
        self.task_id = task_id

    def _append_trace(self, entry: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    def _call_provider_with_budget(
        self,
        *,
        step: int,
        purpose: str,
        messages: list[ChatMessage],
        tools: Sequence[ToolSpec],
        prefix_hash: str,
        temperature: float,
    ) -> tuple[Any, str | None]:
        """Reserve budget, invoke provider (with 1 transient timeout retry), settle usage."""
        for net_attempt in range(2):
            call_purpose = purpose if net_attempt == 0 else f"{purpose}_retry"
            if self.store is not None and self.run_id is not None:
                try:
                    self.store.reserve_llm_call(
                        self.run_id, actor=self.actor, purpose=call_purpose
                    )
                except ValueError as exc:
                    return None, f"budget_exhausted: {exc}"
            try:
                resp = self.provider.complete(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                )
            except Exception as exc:
                if self.store is not None and self.run_id is not None:
                    self.store.settle_llm_call(self.run_id, actor=self.actor, unknown=True)
                self._append_trace(
                    {
                        "timestamp": utc_now().isoformat(),
                        "actor": self.actor,
                        "step": step,
                        "prefix_hash": prefix_hash,
                        "error": str(exc),
                        "retry": net_attempt,
                    }
                )
                is_timeout = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()
                if is_timeout and net_attempt == 0:
                    continue
                return None, f"provider_error: {exc}"

            if self.store is not None and self.run_id is not None:
                self.store.settle_llm_call(
                    self.run_id,
                    actor=self.actor,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                )
            return resp, None
        return None, "provider_error: retry exhausted"

    def complete_repair(
        self,
        *,
        prior_messages: Sequence[ChatMessage],
        correction: str,
        step: int = 1,
        temperature: float = 0.0,
    ) -> LoopResult:
        """Run one tool-free repair turn preserving prior exploration messages."""
        messages = [*prior_messages, ChatMessage(role="user", content=correction)]
        prefix_hash = content_hash({"actor": self.actor, "repair": True})
        resp, err = self._call_provider_with_budget(
            step=step,
            purpose=f"repair_step_{step}",
            messages=messages,
            tools=(),
            prefix_hash=prefix_hash,
            temperature=temperature,
        )
        if resp is None:
            return LoopResult(
                completed=False,
                stop_reason=err or "provider_error",
                final_content="",
                steps_taken=0,
                messages=tuple(messages),
            )
        messages.append(ChatMessage(role="assistant", content=resp.content, tool_calls=()))
        self._append_trace(
            {
                "timestamp": utc_now().isoformat(),
                "actor": self.actor,
                "step": step,
                "prefix_hash": prefix_hash,
                "message_count": len(messages),
                "assistant_content": resp.content,
                "tool_calls": [],
                "tool_outputs": [],
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
            }
        )
        return LoopResult(
            completed=True,
            stop_reason="final_response",
            final_content=resp.content,
            steps_taken=1,
            messages=tuple(messages),
            total_input_tokens=resp.usage.input_tokens,
            total_output_tokens=resp.usage.output_tokens,
        )

    def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        extra_tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
    ) -> LoopResult:
        tool_list: list[ToolSpec] = []
        if self.toolkit is not None:
            tool_list.extend(self.toolkit.tool_specs())
        tool_list.extend(extra_tools)
        frozen_tools = tuple(canonical_tool_specs(tool_list))

        # Static prefix stays invariant across all steps for KV-cache reuse
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        prefix_hash = content_hash(
            {
                "system": system_prompt,
                "user": user_prompt,
                "tools": [t.name for t in frozen_tools],
            }
        )

        total_in = 0
        total_out = 0
        turn_id = new_id("turn")
        turn_events: list[dict[str, Any]] = []

        def _emit(kind: str, payload: dict[str, Any]) -> None:
            turn_events.append({"kind": kind, **payload})
            if self.observer is not None:
                self.observer(kind, payload)

        def _finalize_result(
            *,
            completed: bool,
            stop_reason: str,
            final_content: str,
            steps_taken: int,
            final_messages: list[ChatMessage],
            total_input_tokens: int,
            total_output_tokens: int,
        ) -> LoopResult:
            receipt = build_turn_receipt(
                turn_id=turn_id,
                actor=self.actor,
                task_id=self.task_id,
                events=turn_events,
            )
            _emit("receipt", receipt)
            store = self.store
            run_id = self.run_id
            if store is not None and run_id is not None:
                # Best-effort: a missing run row must never fail an otherwise valid turn.
                with suppress(Exception):
                    store.append_event(
                        run_id=run_id,
                        stage="orchestrate",
                        event_type="turn.receipt",
                        actor=self.actor,
                        payload=receipt,
                    )
            return LoopResult(
                completed=completed,
                stop_reason=stop_reason,
                final_content=final_content,
                steps_taken=steps_taken,
                messages=tuple(final_messages),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                receipt=receipt,
            )

        for step in range(1, self.max_steps + 1):
            if step > 1 and self.trim_seen:
                messages = shrink_seen(messages)
            _emit(
                "step_start",
                {"actor": self.actor, "task_id": self.task_id, "step": step},
            )

            step_tools = frozen_tools
            if (
                self.synthesize_on_max_steps
                and frozen_tools
                and step == self.max_steps
                and self.max_steps > 1
                and any(msg.role == "tool" for msg in messages)
            ):
                step_tools = ()
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "[EXPLORATION BUDGET REACHED] Stop calling tools now and output "
                            "your final response using the repository facts already observed."
                        ),
                    )
                )

            resp, err = self._call_provider_with_budget(
                step=step,
                purpose=f"loop_step_{step}",
                messages=messages,
                tools=step_tools,
                prefix_hash=prefix_hash,
                temperature=temperature,
            )
            if resp is None:
                stop = err or "provider_error"
                steps = step - 1 if stop.startswith("budget_exhausted") else step
                return _finalize_result(
                    completed=False,
                    stop_reason=stop,
                    final_content="",
                    steps_taken=steps,
                    final_messages=messages,
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                )

            total_in += resp.usage.input_tokens
            total_out += resp.usage.output_tokens
            _emit(
                "llm",
                {
                    "actor": self.actor,
                    "task_id": self.task_id,
                    "step": step,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
            )

            # Append-only update: preserve assistant turn + any tool errors
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=resp.content,
                    tool_calls=resp.tool_calls,
                )
            )

            tool_outputs: list[dict[str, str]] = []
            if resp.tool_calls:
                for tc in resp.tool_calls:
                    denied_reason: str | None = None
                    if self.pre_tool_hook is not None:
                        allowed, reason = self.pre_tool_hook(tc.name, tc.arguments)
                        if not allowed:
                            denied_reason = reason
                    if denied_reason is not None:
                        out = f"Error: [PRE_TOOL_HOOK DENIED] {denied_reason}"
                        _emit(
                            "pre_tool_blocked",
                            {
                                "actor": self.actor,
                                "task_id": self.task_id,
                                "step": step,
                                "tool": tc.name,
                                "reason": denied_reason,
                            },
                        )
                    else:
                        _emit(
                            "tool_start",
                            {
                                "actor": self.actor,
                                "task_id": self.task_id,
                                "step": step,
                                "tool": tc.name,
                            },
                        )
                        started = time.monotonic()
                        if self.toolkit is None:
                            out = f"Error: no toolkit attached to execute tool '{tc.name}'."
                        else:
                            out = self.toolkit.execute(tc.name, tc.arguments)
                        elapsed_ms = int((time.monotonic() - started) * 1000)
                        _emit(
                            "tool_end",
                            {
                                "actor": self.actor,
                                "task_id": self.task_id,
                                "step": step,
                                "tool": tc.name,
                                "is_error": out.startswith("Error:"),
                                "duration_ms": elapsed_ms,
                            },
                        )
                    tool_outputs.append({"id": tc.id, "name": tc.name, "output": out})
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=out,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )

            self._append_trace(
                {
                    "timestamp": utc_now().isoformat(),
                    "actor": self.actor,
                    "step": step,
                    "prefix_hash": prefix_hash,
                    "message_count": len(messages),
                    "assistant_content": resp.content,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in resp.tool_calls
                    ],
                    "tool_outputs": tool_outputs,
                    "usage": {
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                    },
                }
            )

            if not resp.tool_calls:
                return _finalize_result(
                    completed=True,
                    stop_reason="final_response",
                    final_content=resp.content,
                    steps_taken=step,
                    final_messages=messages,
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                )

        return _finalize_result(
            completed=False,
            stop_reason="max_steps_exceeded",
            final_content=messages[-1].content if messages else "",
            steps_taken=self.max_steps,
            final_messages=messages,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )
