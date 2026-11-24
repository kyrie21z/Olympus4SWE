"""Free-form human reply routing bounded by deterministic ``RunControl`` actions.

The router is deliberately two-layered:

1. A lightweight LLM pass translates prose into one structured intent.
2. Deterministic sanitisation filters every answer/resolution against the run's
   live artifacts, so a hallucinated or malformed reply can never touch the store.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import ConfigDict

from olympus.core.contracts import CritiqueSeverity, GoalSpec, RunStatus, StrictModel, TaskPlan
from olympus.providers.base import ChatMessage, LLMProvider
from olympus.session.contracts import PauseKind

ReplyAction = Literal["clarify", "approve", "revise", "reopen", "unknown"]

_ROUTER_SYSTEM_PROMPT = """You are the Olympus4SWE Session Reply Router.
Translate one free-form human reply into exactly one routing intent. Never invent
requirements, never answer ambiguities the human did not address, and never mutate state.

Return ONLY one JSON object:
{
  "action": "clarify | approve | revise | reopen | unknown",
  "answers": {"<AMBIGUITY_ID>": "verbatim answer from the human"},
  "resolutions": {"<CRITIQUE_ID>": "how the concern is resolved"},
  "feedback": "revision or follow-up instruction, when the human wants changes",
  "message": "one concise sentence describing the routing decision"
}

Routing rules:
- action=clarify only when the human answers one or more listed ambiguity/critique IDs.
- action=approve only when the human explicitly approves the pending artifact/gate.
- action=revise when the human rejects, asks to change the current plan/spec, or supplies new
  concrete domain facts/constraints/requirements while paused on clarification without answering
  the listed IDs verbatim.
- action=reopen when the run is already completed and the human asks for follow-up work.
- action=unknown when the reply is ambiguous, a pure refusal, or a question about defaults;
  do not guess.
- Use only the IDs listed in the context; omit answers for anything not addressed."""


class ReplyIntent(StrictModel):
    """A deterministic, sanitised routing decision."""

    model_config = ConfigDict(extra="ignore")

    action: ReplyAction = "unknown"
    answers: dict[str, str] = {}
    resolutions: dict[str, str] = {}
    feedback: str = ""
    message: str = ""


class _RawIntent(ReplyIntent):
    model_config = ConfigDict(extra="ignore")


_APPROVE_TOKENS = (
    "approve",
    "approved",
    "lgtm",
    "looks good",
    "go ahead",
    "proceed",
    "ship it",
    "同意",
    "批准",
    "通过",
    "没问题",
    "继续",
)
_REVISE_TOKENS = (
    "revise",
    "rework",
    "change",
    "instead",
    "reject",
    "not right",
    "redo",
    "修改",
    "调整",
    "重做",
    "不对",
    "改一下",
)


def _blocking_ambiguity_ids(spec: GoalSpec | None) -> set[str]:
    if spec is None:
        return set()
    return {amb.id for amb in spec.open_ambiguities if amb.blocking and not amb.answer}


def _blocking_critique_ids(spec: GoalSpec | None) -> set[str]:
    if spec is None:
        return set()
    return {
        note.id
        for note in spec.minority_notes
        if note.severity == CritiqueSeverity.BLOCKING and not note.resolved
    }


def _reply_context(
    *,
    message: str,
    snapshot: dict[str, Any],
    pause_kind: PauseKind | None,
    gate: str | None,
    completed: bool,
) -> str:
    spec: GoalSpec | None = snapshot.get("spec")
    plan: TaskPlan | None = snapshot.get("plan")
    parts: list[str] = [
        f"Run status: {snapshot['status'].value}",
        f"Pause kind: {pause_kind.value if pause_kind else 'none'}",
        f"Pending gate: {gate or 'none'}",
        f"Run already completed: {str(completed).lower()}",
    ]
    if spec is not None:
        parts.append(f"GoalSpec goal: {spec.goal}")
        ambiguity_lines: list[str] = []
        for amb in spec.open_ambiguities:
            if not amb.blocking or amb.answer:
                continue
            options = (
                f" options={json.dumps(amb.options, ensure_ascii=False)}" if amb.options else ""
            )
            ambiguity_lines.append(f"- {amb.id}: {amb.question}{options}")
        if ambiguity_lines:
            parts.append("Open blocking ambiguities:\n" + "\n".join(ambiguity_lines))
        critique_lines = [
            f"- {note.id}: {note.summary}"
            for note in spec.minority_notes
            if note.severity == CritiqueSeverity.BLOCKING and not note.resolved
        ]
        if critique_lines:
            parts.append("Unresolved blocking critiques:\n" + "\n".join(critique_lines))
    if plan is not None:
        tasks = "; ".join(f"{task.task_id}={task.title}" for task in plan.tasks)
        parts.append(f"Current TaskPlan tasks: {tasks or '(none)'}")
    parts.append(f"Human reply:\n{message}")
    return "\n".join(parts)


_NEGATED_APPROVAL_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:do\s+not|don'?t|cannot|can'?t|won'?t|will\s+not|shouldn'?t|should\s+not|wouldn'?t|never)\b[^.!?\n]{0,40}\b(?:proceed|approve|approved|approving|go\s+ahead|continue|ship|merge)\b",
        r"\bnot\b[^.!?,;\n]{0,40}\b(?:proceed|approve|approved|approving|go\s+ahead|continue|ship|merge)\b",
        r"\bnot\s+(?:ready|approv(?:e|ed|al)|good\s+to\s+go|ok(?:ay)?|yet)\b",
        r"\b(?:not|never)\s+approv(?:e|ed|al)\b",
        r"\bhold\s+off\b",
        r"\bno\s*[,;:]",
        r"^\s*no\b",
        r"\breject(?:s|ed|ing)?\b",
        r"\bcancel(?:s|led|ing)?\b",
    )
)
_NEGATED_APPROVAL_CJK = ("不要", "不批准", "不同意", "拒绝", "取消", "暂不", "先不")
# ``别`` negates an instruction, but is also a morpheme in 特别/分别/区别/个别 ….
_NEGATED_APPROVAL_CJK_BIE = re.compile(r"(?<![特分区个性差级告])别")
_NEGATED_REVISION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:do\s+not|don'?t|cannot|can'?t|won'?t|will\s+not|shouldn'?t|should\s+not|wouldn'?t|never|no\s+need\s+to|stop)\b[^.!?,;\n]{0,40}\b(?:change|modify|edit|revise|rework|redo|touch|reopen|alter|update)\b",
        r"\bno\s+(?:changes?|modifications?|edits?|revisions?|further\s+work)\b",
        r"\bleave\s+(?:it|everything|this|that)\s+as\s+is\b",
        r"\bkeep\s+(?:it|everything|this|that|all)\s+(?:as\s+is|unchanged|the\s+same)\b",
        r"\bstay\s+(?:as\s+is|unchanged)\b",
        r"\bnothing\s+(?:else\s+)?to\s+(?:change|fix|do|modify|update)\b",
    )
)
_NEGATED_REVISION_CJK = (
    "不要改",
    "别改",
    "不用改",
    "不要修改",
    "不用修改",
    "无需修改",
    "保持原样",
    "不要动",
)
_EXPLICIT_ANSWER_LINE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:(?:ambiguity|critique)\s*:\s*)?([A-Za-z][A-Za-z0-9_-]*)\s*[:=]\s*(.+?)\s*$"
)
_ACTIONABLE_CHARS = re.compile(r"[0-9A-Za-z\u3400-\u9fff]")

_DEFAULT_ACCEPTANCE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:use|using|accept|accepting|apply|applying|take|taking|adopt|adopting|"
        r"go\s+with|going\s+with|proceed\s+with|proceeding\s+with|follow|following|"
        r"stick\s+with|sticking\s+with|ok\s+with|okay\s+with|fine\s+with)\b[^.!?\n]{0,60}"
        r"\b(?:defaults?|default\s+assumptions?|recommendations?|recommended\s+"
        r"(?:answers?|options?|choices?|defaults?|assumptions?)|"
        r"your\s+(?:best\s+)?(?:defaults?|recommendations?|assumptions?))\b",
        r"\b(?:go\s+with|proceed\s+with|stick\s+with)\s+(?:the\s+|your\s+)?"
        r"(?:defaults?|recommendations?)\b",
        r"^\s*(?:the\s+|your\s+)?(?:defaults?|default\s+(?:assumptions?|answers?|choices?)|"
        r"recommendations?)\s*[.!]?\s*$",
        r"^\s*(?:please\s+)?(?:go\s+on|go\s+ahead|continue|proceed|carry\s+on|keep\s+going)\s*[.!]?\s*$",
    )
)
_QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:what|which|why|how|who|when|where|should\s+we|can\s+you\s+explain|"
    r"could\s+you\s+explain|tell\s+me\s+what|list\s+(?:the|your)|show\s+me)\b"
)
_CONDITIONAL_DEFAULTS_RE = re.compile(
    r"\b(?:if|unless|whether|suppose|supposing|what\s+if|in\s+case)\b[^.!?\n]{0,60}"
    r"\b(?:defaults?|assumptions?|recommendations?)\b"
)
_DEFAULT_ACCEPTANCE_CJK = (
    "按默认",
    "按推荐",
    "按你的默认",
    "按你的推荐",
    "按照默认",
    "按照推荐",
    "用默认",
    "用推荐",
    "用你的默认",
    "用你的推荐",
    "采用默认",
    "采用推荐",
    "使用默认",
    "使用推荐",
    "默认方案继续",
    "默认方案推进",
    "推荐方案继续",
    "推荐方案推进",
)
_NEGATED_DEFAULTS_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:do\s+not|don'?t|never|cannot|can'?t|won'?t|will\s+not|shouldn'?t|"
        r"should\s+not|no\s+need\s+to|stop)\b[^.!?\n]{0,60}\b(?:defaults?|assumptions?|"
        r"recommendations?)\b",
        r"\bnot\s+(?:your\s+)?(?:defaults?|recommendations?|assumptions?)\b",
        r"\bno\s+defaults?\b",
    )
)
_NEGATED_DEFAULTS_CJK = (
    "不要默认",
    "不要按默认",
    "不要用默认",
    "不要采用默认",
    "不要使用默认",
    "别按默认",
    "别用默认",
    "不用默认",
    "不要按推荐",
    "不要用推荐",
    "别按推荐",
    "不采用推荐",
    "不要你的默认",
    "不用你的推荐",
    "不按默认",
    "不按推荐",
)


def _has_actionable_text(text: str) -> bool:
    return len(_ACTIONABLE_CHARS.findall(text)) >= 3


def _negated_defaults(text: str) -> bool:
    if any(pattern.search(text) for pattern in _NEGATED_DEFAULTS_PATTERNS):
        return True
    return any(token in text for token in _NEGATED_DEFAULTS_CJK)


def _pending_blockers(snapshot: dict[str, Any] | None) -> bool:
    if not snapshot:
        return False
    spec: GoalSpec | None = snapshot.get("spec")
    return bool(_blocking_ambiguity_ids(spec) or _blocking_critique_ids(spec))


def _default_blocker_answers(
    snapshot: dict[str, Any] | None, message: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Fill every pending blocker deterministically for a "use your defaults" reply."""
    spec: GoalSpec | None = snapshot.get("spec") if snapshot else None
    if spec is None:
        return {}, {}
    fallback = message.strip()
    answers: dict[str, str] = {}
    for amb in spec.open_ambiguities:
        if not amb.blocking or amb.answer:
            continue
        choice = (amb.default_assumption or "").strip()
        if not choice and amb.options:
            choice = next((item.strip() for item in amb.options if item.strip()), "")
        if not choice:
            choice = fallback
        if choice:
            answers[amb.id] = choice
    resolutions: dict[str, str] = {}
    for note in spec.minority_notes:
        if note.severity != CritiqueSeverity.BLOCKING or note.resolved:
            continue
        if fallback:
            resolutions[note.id] = fallback
    return answers, resolutions


def _is_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(("?", "？")):
        return True
    return _QUESTION_PREFIX_RE.search(stripped.casefold()) is not None


def _is_default_delegation(text: str) -> bool:
    """True when the text affirmatively delegates decisions to recorded defaults."""
    lowered = text.casefold()
    if (
        _is_question(text)
        or _CONDITIONAL_DEFAULTS_RE.search(lowered)
        or _negated_approval(lowered)
        or _negated_revision(lowered)
        or _negated_defaults(lowered)
    ):
        return False
    return any(pattern.search(lowered) for pattern in _DEFAULT_ACCEPTANCE_PATTERNS) or any(
        token in lowered for token in _DEFAULT_ACCEPTANCE_CJK
    )


def _default_acceptance(
    text: str,
    *,
    snapshot: dict[str, Any] | None,
    message: str,
) -> tuple[dict[str, str], dict[str, str]] | None:
    """Return deterministic default answers when the human delegates to defaults."""
    if not _pending_blockers(snapshot):
        return None
    if not _is_default_delegation(text):
        return None
    return _default_blocker_answers(snapshot, message)


def _negated_approval(text: str) -> bool:
    if any(pattern.search(text) for pattern in _NEGATED_APPROVAL_PATTERNS):
        return True
    if any(token in text for token in _NEGATED_APPROVAL_CJK):
        return True
    return _NEGATED_APPROVAL_CJK_BIE.search(text) is not None


def _negated_revision(text: str) -> bool:
    if any(pattern.search(text) for pattern in _NEGATED_REVISION_PATTERNS):
        return True
    return any(token in text for token in _NEGATED_REVISION_CJK)


def _extract_explicit_clarifications(
    message: str, snapshot: dict[str, Any] | None
) -> tuple[dict[str, str], dict[str, str]]:
    if not snapshot:
        return {}, {}
    spec: GoalSpec | None = snapshot.get("spec")
    valid_answers = {item.casefold(): item for item in _blocking_ambiguity_ids(spec)}
    valid_resolutions = {item.casefold(): item for item in _blocking_critique_ids(spec)}
    if not valid_answers and not valid_resolutions:
        return {}, {}
    answers: dict[str, str] = {}
    resolutions: dict[str, str] = {}
    for line in message.splitlines():
        match = _EXPLICIT_ANSWER_LINE.match(line)
        if not match:
            continue
        raw_id, value = match.group(1).strip(), match.group(2).strip()
        if not value:
            continue
        key = raw_id.casefold()
        if key in valid_answers:
            answers[valid_answers[key]] = value
        elif key in valid_resolutions:
            resolutions[valid_resolutions[key]] = value
    return answers, resolutions


def _merge_default_and_explicit(
    defaults: tuple[dict[str, str], dict[str, str]],
    explicit_answers: dict[str, str],
    explicit_resolutions: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Merge deterministic defaults with explicit per-ID answers (explicit text wins)."""
    default_answers, default_resolutions = defaults
    default_answers.update(
        {
            key: value
            for key, value in explicit_answers.items()
            if not _is_default_delegation(value)
        }
    )
    default_resolutions.update(
        {
            key: value
            for key, value in explicit_resolutions.items()
            if not _is_default_delegation(value)
        }
    )
    return default_answers, default_resolutions


def _is_approval_question(message: str, lowered: str) -> bool:
    """True for pure confirmation/proceed questions that must never trigger execution."""
    return _is_question(message) and any(token in lowered for token in _APPROVE_TOKENS)


def _keyword_fallback(
    message: str,
    *,
    snapshot: dict[str, Any] | None = None,
    pause_kind: PauseKind | None = None,
) -> ReplyIntent:
    """Conservative fallback used only when the structured parse fails."""
    text = message.casefold().strip()
    if not text:
        return ReplyIntent(action="unknown", message="Empty reply")
    answers, resolutions = _extract_explicit_clarifications(message, snapshot)
    defaults = _default_acceptance(text, snapshot=snapshot, message=message)
    if defaults is not None:
        answers, resolutions = _merge_default_and_explicit(defaults, answers, resolutions)
    if answers or resolutions:
        return ReplyIntent(
            action="clarify",
            answers=answers,
            resolutions=resolutions,
            message="Keyword fallback: explicit blocker answers",
        )
    revise = (
        any(token in text for token in _REVISE_TOKENS)
        and not _negated_revision(text)
        and not _CONDITIONAL_DEFAULTS_RE.search(text)
    )
    if _negated_approval(text):
        # A negated approval is never an approval: treat an explicit revision ask
        # as a revision and everything else as ambiguous, so no gate is confirmed.
        if revise:
            return ReplyIntent(
                action="revise",
                feedback=message.strip(),
                message="Keyword fallback: negated approval, treated as revision",
            )
        return ReplyIntent(
            action="unknown",
            message="Keyword fallback: negated approval needs an explicit revision request",
        )
    if revise:
        return ReplyIntent(
            action="revise", feedback=message.strip(), message="Keyword fallback: revision"
        )
    if (
        pause_kind == PauseKind.BLOCKED
        and _has_actionable_text(message)
        and not _is_approval_question(message, text)
        and not (_negated_revision(text) or _negated_defaults(text))
    ):
        # Any concrete hint (path, symbol, diagnosis, redirect) is actionable recovery
        # context while blocked; never deadlock on an unmappable reply.
        return ReplyIntent(
            action="revise",
            feedback=message.strip(),
            message="Keyword fallback: blocked recovery hint",
        )
    if (
        any(token in text for token in _APPROVE_TOKENS)
        and not _is_question(message)
        and not _negated_revision(text)
        and not _negated_defaults(text)
    ):
        return ReplyIntent(action="approve", message="Keyword fallback: approval")
    return ReplyIntent(action="unknown", message="Could not parse the reply safely")


def _sanitize(
    raw: ReplyIntent,
    *,
    snapshot: dict[str, Any],
    message: str,
    pause_kind: PauseKind | None = None,
) -> ReplyIntent:
    spec: GoalSpec | None = snapshot.get("spec")
    valid_answers = _blocking_ambiguity_ids(spec)
    valid_resolutions = _blocking_critique_ids(spec)
    answers = {
        key.strip(): value.strip()
        for key, value in raw.answers.items()
        if key.strip() in valid_answers and value.strip()
    }
    resolutions = {
        key.strip(): value.strip()
        for key, value in raw.resolutions.items()
        if key.strip() in valid_resolutions and value.strip()
    }
    explicit_answers, explicit_resolutions = _extract_explicit_clarifications(message, snapshot)
    lowered = message.casefold()
    asks_about_defaults = _is_question(message) and any(
        token in lowered for token in ("default", "assumption", "recommend", "默认", "推荐")
    )
    if (
        _negated_defaults(lowered)
        or _negated_revision(lowered)
        or _CONDITIONAL_DEFAULTS_RE.search(lowered)
        or asks_about_defaults
    ):
        # Never accept LLM-guessed answers when the user rejected defaults, asked a
        # question about defaults, or posed a conditional warning; keep only explicit IDs.
        answers = dict(explicit_answers)
        resolutions = dict(explicit_resolutions)
    else:
        answers.update(explicit_answers)
        resolutions.update(explicit_resolutions)
    defaults = _default_acceptance(lowered, snapshot=snapshot, message=message)
    if defaults is not None:
        # A human delegation to defaults resolves every remaining blocker
        # deterministically; explicit per-ID text in the message still wins.
        answers, resolutions = _merge_default_and_explicit(
            defaults, explicit_answers, explicit_resolutions
        )
    action = raw.action
    feedback = raw.feedback.strip()
    if answers or resolutions:
        # Human answers always win: never approve/revise while silently dropping them.
        action = "clarify"
    elif action == "clarify":
        action = "unknown"
    if action == "approve" and (
        _negated_approval(lowered) or _negated_defaults(lowered) or _is_question(message)
    ):
        revise = any(token in lowered for token in _REVISE_TOKENS) and not _negated_revision(
            lowered
        )
        action = "revise" if revise else "unknown"
        feedback = message.strip() if revise else ""
    elif (
        action == "approve"
        and _negated_revision(lowered)
        and not any(token in lowered for token in _APPROVE_TOKENS)
    ):
        # A pure "do not change anything" reply is never an approval.
        action = "unknown"
        feedback = ""
    if action == "reopen" and (_negated_approval(lowered) or _negated_defaults(lowered)):
        # A refusal must never reopen a completed run, even if the router guessed so.
        action = "unknown"
        feedback = ""
    if action in ("reopen", "revise") and (
        _negated_revision(lowered)
        or not _has_actionable_text(message)
        or _is_approval_question(message, lowered)
    ):
        action = "unknown"
        feedback = ""
    if action == "revise" and not feedback:
        feedback = message.strip()
        if not feedback:
            action = "unknown"
    if (
        action in ("unknown", "approve", "reopen")
        and pause_kind == PauseKind.BLOCKED
        and _has_actionable_text(message)
        and not _is_approval_question(message, lowered)
        and not _negated_revision(lowered)
        and not _negated_approval(lowered)
        and not _negated_defaults(lowered)
    ):
        # While blocked, any concrete hint is recovery context: route it to revision
        # instead of approving a gate the deterministic validator already rejected.
        action = "revise"
        feedback = message.strip()
    return ReplyIntent(
        action=action,
        answers=answers,
        resolutions=resolutions,
        feedback=feedback,
        message=raw.message.strip(),
    )


def _fast_path_intent(
    *,
    message: str,
    snapshot: dict[str, Any],
    pause_kind: PauseKind | None,
    completed: bool,
) -> ReplyIntent | None:
    """Resolve unambiguous replies deterministically without an extra LLM round-trip."""
    if completed:
        return None
    lowered = message.casefold().strip()
    if not lowered:
        return None
    if _pending_blockers(snapshot):
        explicit_answers, explicit_resolutions = _extract_explicit_clarifications(
            message, snapshot
        )
        defaults = _default_acceptance(lowered, snapshot=snapshot, message=message)
        if defaults is not None:
            merged_answers, merged_resolutions = _merge_default_and_explicit(
                defaults, explicit_answers, explicit_resolutions
            )
            if merged_answers or merged_resolutions:
                return ReplyIntent(
                    action="clarify",
                    answers=merged_answers,
                    resolutions=merged_resolutions,
                    message="Deterministic fast-path: default assumptions accepted",
                )
        spec: GoalSpec | None = snapshot.get("spec")
        needed_answers = _blocking_ambiguity_ids(spec)
        needed_resolutions = _blocking_critique_ids(spec)
        if (
            (explicit_answers or explicit_resolutions)
            and needed_answers.issubset(explicit_answers)
            and needed_resolutions.issubset(explicit_resolutions)
        ):
            return ReplyIntent(
                action="clarify",
                answers=explicit_answers,
                resolutions=explicit_resolutions,
                message="Deterministic fast-path: all pending blockers answered",
            )
        if explicit_answers or explicit_resolutions:
            # Partial explicit answers must flow through LLM + sanitizer rather than
            # being swallowed as a generic blocked-recovery revision.
            return None
    if (
        (pause_kind == PauseKind.BLOCKED or snapshot.get("status") == RunStatus.FAILED)
        and _has_actionable_text(message)
        and not _is_question(message)
        and not (
            _negated_revision(lowered)
            or _negated_approval(lowered)
            or _negated_defaults(lowered)
        )
    ):
        return ReplyIntent(
            action="revise",
            feedback=message.strip(),
            message="Deterministic fast-path: blocked recovery hint",
        )
    return None


_ROUTER_TIMEOUT_SECONDS = 30.0


def _call_router_provider(
    provider: LLMProvider,
    messages: list[ChatMessage],
) -> Any:
    """Invoke ``provider.complete`` with fast non-thinking classification hints when supported."""
    import inspect

    kwargs: dict[str, Any] = {"messages": messages}
    params: dict[str, inspect.Parameter]
    try:
        params = dict(inspect.signature(provider.complete).parameters)
    except (TypeError, ValueError):
        params = {}
    accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_varkw or "enable_thinking" in params:
        kwargs["enable_thinking"] = False
    if accepts_varkw or "stream" in params:
        kwargs["stream"] = True
    if accepts_varkw or "timeout" in params:
        kwargs["timeout"] = _ROUTER_TIMEOUT_SECONDS
    return provider.complete(**kwargs)


def parse_reply(
    *,
    provider: LLMProvider,
    message: str,
    snapshot: dict[str, Any],
    pause_kind: PauseKind | None,
    gate: str | None = None,
    completed: bool = False,
) -> ReplyIntent:
    """Translate prose into a sanitised intent; malformed output degrades safely."""
    fast = _fast_path_intent(
        message=message,
        snapshot=snapshot,
        pause_kind=pause_kind,
        completed=completed,
    )
    if fast is not None:
        return fast
    user_prompt = _reply_context(
        message=message,
        snapshot=snapshot,
        pause_kind=pause_kind,
        gate=gate,
        completed=completed,
    )
    response = _call_router_provider(
        provider,
        [
            ChatMessage(role="system", content=_ROUTER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ],
    )
    try:
        parsed = _RawIntent.model_validate_json(_extract_json(response.content))
    except Exception:  # noqa: BLE001 -- malformed router JSON falls back to keyword guard
        return _keyword_fallback(message, snapshot=snapshot, pause_kind=pause_kind)
    intent = _sanitize(parsed, snapshot=snapshot, message=message, pause_kind=pause_kind)
    if intent.action == "unknown":
        return _keyword_fallback(message, snapshot=snapshot, pause_kind=pause_kind)
    return intent


def _extract_json(content: str) -> str:
    from olympus.harness.loop import extract_json_payload

    return extract_json_payload(content)


__all__ = ["ReplyIntent", "parse_reply"]
