"""Shared read-only structured-output runner for repository planning stages.

Native callers keep ``AgentLoop.run_structured`` semantics unchanged. External
``AgentDriver`` turns (e.g. ``PiDriver``) run with ``read_only=True`` on the
workspace root, validate the returned JSON with Pydantic plus the caller's
zero-token gate, and feed violations back through a bounded retry budget. The
control plane, never the agent, persists the resulting artifact.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from olympus.core.contracts import BudgetLedger, BudgetPolicy
from olympus.core.store import RunStore
from olympus.harness.driver import AgentDriver, DriverEvent, DriverTurnRequest, TurnOutcome
from olympus.harness.loop import AgentLoop, parse_model_json


class StructuredTurnError(RuntimeError):
    """Raised when an external driver stops before returning a usable payload."""


def _validation_prompt(
    user_prompt: str,
    header: str,
    violations: Sequence[str],
    instruction: str,
    *,
    previous_output: str | None = None,
) -> str:
    draft = (previous_output or "").strip()
    if draft:
        snippet = draft if len(draft) <= 16000 else draft[:16000] + "\n...[truncated]"
        return (
            f"{user_prompt}\n\n"
            f"Previous draft output (preserve grounded facts; fix ONLY the errors below "
            f"without re-exploring the repository):\n{snippet}\n\n"
            f"{header}\nViolations: {'; '.join(violations)}\n{instruction}"
        )
    return f"{user_prompt}\n\n{header}\nViolations: {'; '.join(violations)}\n{instruction}"


def _is_non_retryable_stop_reason(reason: str) -> bool:
    lowered = reason.lower()
    return any(
        token in lowered
        for token in (
            "cancel",
            "interrupt",
            "abort",
            "budget",
            "timeout",
            "timed_out",
            "timed out",
            "deadline",
        )
    )


def _settle_turn_usage(
    *,
    driver: AgentDriver,
    outcome: TurnOutcome,
    store: RunStore | None,
    run_id: str | None,
    actor: str,
    attempt: int,
) -> None:
    """Book one external turn unless the driver already settled per-step usage."""
    if store is None or run_id is None:
        return
    if getattr(driver, "records_usage_internally", False):
        return
    store.record_external_llm_usage(
        run_id,
        actor=actor,
        purpose=f"{driver.name}_structured_{attempt}",
        input_tokens=outcome.total_input_tokens,
        output_tokens=outcome.total_output_tokens,
    )


def run_structured_turn[T: BaseModel](
    *,
    loop: AgentLoop,
    driver: AgentDriver | None,
    driver_model: str | None,
    store: RunStore | None,
    run_id: str | None,
    workspace_root: Path,
    actor: str,
    system_prompt: str,
    user_prompt: str,
    model_cls: type[T],
    trace_path: Path | None = None,
    max_steps: int | None = None,
    max_parse_retries: int = 2,
    temperature: float = 0.0,
    max_validation_retries: int = 0,
    validation_header: str = "[VALIDATION FAILED]",
    validation_instruction: str = "Fix all violations and return valid JSON.",
    transform: Callable[[T], T] | None = None,
    validator: Callable[[T], Sequence[str]] | None = None,
    on_validation_exhausted: Callable[[T, list[str]], T] | None = None,
    on_event: DriverEvent | None = None,
) -> T:
    """Return one validated payload from the native loop or an external driver."""
    if driver is None:
        return loop.run_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_cls=model_cls,
            max_parse_retries=max_parse_retries,
            temperature=temperature,
            transform=transform,
            validator=validator,
            max_validation_retries=max_validation_retries,
            validation_header=validation_header,
            validation_instruction=validation_instruction,
            on_validation_exhausted=on_validation_exhausted,
        )

    prompt = user_prompt
    turn_max_steps = max_steps
    turn_disable_tools = False
    disable_tools_supported = True
    synthesized_after_cap = False
    parse_retries = 0
    validation_retries = 0
    attempt = 0
    while True:
        attempt += 1
        if store is not None and run_id is not None:
            snap = store.get_run(run_id)
            ledger = snap.get("budget_ledger")
            policy = snap.get("budget_policy")
            if (
                isinstance(ledger, BudgetLedger)
                and isinstance(policy, BudgetPolicy)
                and not ledger.can_reserve(policy)
            ):
                raise StructuredTurnError(
                    f"{driver.name} driver stopped before attempt {attempt} for "
                    f"{model_cls.__name__}: budget_exhausted"
                )
        outcome = driver.run_turn(
            DriverTurnRequest(
                workspace_path=workspace_root,
                env_root=workspace_root,
                system_prompt=system_prompt,
                user_prompt=prompt,
                actor=actor,
                read_only=True,
                disable_tools=turn_disable_tools,
                max_steps=turn_max_steps,
                trace_path=trace_path,
                model=driver_model,
            ),
            on_event=on_event,
        )
        _settle_turn_usage(
            driver=driver,
            outcome=outcome,
            store=store,
            run_id=run_id,
            actor=actor,
            attempt=attempt,
        )
        if not outcome.completed:
            if (
                turn_disable_tools
                and disable_tools_supported
                and outcome.steps_taken == 0
                and outcome.total_output_tokens == 0
                and not _is_non_retryable_stop_reason(outcome.stop_reason)
            ):
                # Some external driver backends (e.g., hosted free-tier CLIs) reject
                # zero-tool requests before running a step; fall back once to strict
                # read_only tools while preserving the no-tool prompt & budget guard.
                disable_tools_supported = False
                turn_disable_tools = False
                continue
            if outcome.stop_reason == "max_steps_exceeded" and not synthesized_after_cap:
                synthesized_after_cap = True
                prompt = _validation_prompt(
                    user_prompt,
                    "[STEP LIMIT REACHED]",
                    ["exploration reached the step budget before returning JSON"],
                    "Tools are disabled. Synthesize and return ONLY a single raw JSON object "
                    f"conforming to {model_cls.__name__} from the observed workspace context.",
                    previous_output=outcome.final_content,
                )
                turn_disable_tools = disable_tools_supported
                turn_max_steps = 2
                continue
            raise StructuredTurnError(
                f"{driver.name} driver stopped before returning {model_cls.__name__}: "
                f"{outcome.stop_reason}"
            )
        try:
            candidate = parse_model_json(outcome.final_content, model_cls)
        except (json.JSONDecodeError, ValidationError) as exc:
            violations = [f"output is not valid {model_cls.__name__} JSON: {exc}"]
            if parse_retries < max_parse_retries:
                parse_retries += 1
                prompt = _validation_prompt(
                    user_prompt,
                    validation_header,
                    violations,
                    "Do NOT call any tools. Fix ONLY the JSON/schema errors in the previous "
                    "draft (ensure any provenance or reference list copies exact strings from "
                    "its source list, or is empty) and output a single raw JSON object.",
                    previous_output=outcome.final_content,
                )
                turn_disable_tools = disable_tools_supported
                turn_max_steps = 2
                continue
            raise ValueError(
                f"Failed to parse {model_cls.__name__} JSON from {driver.name}: {exc}"
            ) from exc
        if transform is not None:
            candidate = transform(candidate)
        violations = list(validator(candidate)) if validator is not None else []
        if not violations:
            return candidate
        if validation_retries < max_validation_retries:
            validation_retries += 1
            prompt = _validation_prompt(
                user_prompt,
                validation_header,
                violations,
                validation_instruction,
                previous_output=candidate.model_dump_json(indent=2),
            )
            turn_disable_tools = disable_tools_supported
            turn_max_steps = 2
            continue
        if on_validation_exhausted is not None:
            return on_validation_exhausted(candidate, violations)
        raise ValueError(
            f"{model_cls.__name__} failed validation: {'; '.join(violations)}"
        )


__all__ = ["StructuredTurnError", "run_structured_turn"]
