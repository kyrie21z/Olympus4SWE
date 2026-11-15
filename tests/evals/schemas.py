"""Typed contracts for the Stage 1 four-class evaluation benchmark.

The schemas here are deliberately separate from ``olympus.core.contracts``:
they describe benchmark fixtures, oracle replies, rubric metrics, and suite
reports, while the production contracts stay untouched. Both layers share
``StrictModel`` so unknown JSON fields fail loudly instead of being dropped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, ValidationError, field_validator

from olympus.core.contracts import StrictModel

# ---------------------------------------------------------------------------
# Benchmark fixtures
# ---------------------------------------------------------------------------


class BenchmarkClass(StrEnum):
    """Ambiguity/difficulty class of a Stage 1 evaluation case."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class HiddenSlot(StrictModel):
    """One factual detail the oracle only reveals when asked about it."""

    slot_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    answer: str = Field(min_length=1)
    forbidden_leak_terms: list[str] = Field(default_factory=list)
    blocking: bool = True


class RepoFileFixture(StrictModel):
    """A file materialized into the isolated evaluation repository."""

    path: str = Field(min_length=1)
    content: str

    @field_validator("path")
    @classmethod
    def _validate_repo_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError(f"repo fixture paths must use POSIX separators: {value}")
        pure = PurePosixPath(value)
        if pure.is_absolute():
            raise ValueError(f"repo fixture path must be relative: {value}")
        if ".." in pure.parts:
            raise ValueError(f"repo fixture path cannot escape the repo: {value}")
        return value


class ExpectedCriteria(StrictModel):
    """Deterministic per-case acceptance rubric."""

    expect_refusal_or_pause: bool = False
    min_ambiguity_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    max_questions: int = Field(default=4, ge=0)
    max_clarify_rounds: int = Field(default=3, ge=0)
    require_repo_conflict_check: bool = False
    expected_repo_conflict_terms: list[str] = Field(default_factory=list)
    required_spec_terms: list[str] = Field(default_factory=list)
    require_confirmable_at_end: bool = True
    require_downstream_plan_valid: bool = False
    acceptance_test_command: str | None = None
    leak_check: bool = True


class EvalCase(StrictModel):
    """A single benchmark row loaded from ``datasets/*.jsonl``."""

    case_id: str = Field(min_length=1)
    benchmark_class: BenchmarkClass
    title: str = Field(min_length=1)
    raw_goal: str = Field(min_length=1)
    initial_constraints: list[str] = Field(default_factory=list)
    hidden_slots: list[HiddenSlot] = Field(default_factory=list)
    repo_files: list[RepoFileFixture] = Field(default_factory=list)
    expected: ExpectedCriteria = Field(default_factory=ExpectedCriteria)
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Oracle, judge, and rubric results
# ---------------------------------------------------------------------------


class OracleReply(StrictModel):
    """The oracle's answer to one Stage 1 clarification question."""

    question: str = Field(min_length=1)
    matched_slot_id: str | None = None
    answer: str = Field(min_length=1)
    revealed: bool = False
    leakage_detected: bool = False
    leak_terms: list[str] = Field(default_factory=list)


class RubricMetrics(StrictModel):
    """Every rubric component kept separately for auditability."""

    bounded_scope: float = Field(ge=0.0, le=1.0)
    unresolved_blockers_ok: float = Field(ge=0.0, le=1.0)
    unresolved_blockers_count: int = Field(ge=0)
    failure_modes: float = Field(ge=0.0, le=1.0)
    falsifiable_criteria: float = Field(ge=0.0, le=1.0)
    downstream_consistency: float = Field(ge=0.0, le=1.0)
    vague_goal_refusal: float = Field(ge=0.0, le=1.0)
    ambiguity_recall: float = Field(ge=0.0, le=1.0)
    ask_precision: float = Field(ge=0.0, le=1.0)
    over_asking_penalty: float = Field(ge=0.0, le=1.0)
    leakage_detected: bool
    bounded_turns_ok: float = Field(ge=0.0, le=1.0)
    tkqr: float = Field(default=1.0, ge=0.0, le=1.0)
    ora: float = Field(default=1.0, ge=0.0, le=1.0)
    total_score: float = Field(ge=0.0, le=1.0)
    questions_asked: int = Field(ge=0)
    clarify_rounds: int = Field(ge=0)
    repeated_question_rate: float = Field(ge=0.0, le=1.0)
    fabricated_paths: list[str] = Field(default_factory=list)
    leak_terms: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class JudgeResult(StrictModel):
    """LLM-as-judge outcome; parse failures are ``unscored``, never zero."""

    status: Literal["scored", "unscored", "skipped"] = "skipped"
    met: bool | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = ""


class CaseRunRecord(StrictModel):
    """One (case, variant, trial) execution record."""

    case_id: str = Field(min_length=1)
    benchmark_class: BenchmarkClass
    variant: str = Field(min_length=1)
    trial: int = Field(ge=1)
    passed: bool
    total_score: float = Field(ge=0.0, le=1.0)
    metrics: RubricMetrics
    judge: JudgeResult = Field(
        default_factory=lambda: JudgeResult(status="skipped", reason="judge disabled")
    )
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    error: str | None = None


# ---------------------------------------------------------------------------
# Scorecards and suite reports
# ---------------------------------------------------------------------------


class ClassScorecard(StrictModel):
    """Aggregated variant metrics for one benchmark class."""

    cases: int = Field(ge=0)
    records: int = Field(ge=0)
    passed: int = Field(ge=0)
    pass_at_1: float = Field(ge=0.0, le=1.0)
    pass_all: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_total_score: float = Field(ge=0.0, le=1.0)
    ambiguity_recall: float = Field(ge=0.0, le=1.0)
    ask_precision: float = Field(ge=0.0, le=1.0)
    over_asking_penalty: float = Field(ge=0.0, le=1.0)
    leakage_rate: float = Field(ge=0.0, le=1.0)
    avg_questions: float = Field(ge=0.0)
    avg_rounds: float = Field(ge=0.0)
    tkqr: float = Field(default=0.0, ge=0.0, le=1.0)
    ora: float = Field(default=0.0, ge=0.0, le=1.0)


class VariantScorecard(StrictModel):
    """Aggregated metrics for one harness variant (e.g. baseline/critique)."""

    variant: str = Field(min_length=1)
    cases: int = Field(ge=0)
    records: int = Field(ge=0)
    passed: int = Field(ge=0)
    pass_at_1: float = Field(ge=0.0, le=1.0)
    pass_all: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_total_score: float = Field(ge=0.0, le=1.0)
    ambiguity_recall: float = Field(ge=0.0, le=1.0)
    ask_precision: float = Field(ge=0.0, le=1.0)
    over_asking_penalty: float = Field(ge=0.0, le=1.0)
    leakage_rate: float = Field(ge=0.0, le=1.0)
    avg_questions: float = Field(ge=0.0)
    avg_rounds: float = Field(ge=0.0)
    tkqr: float = Field(default=0.0, ge=0.0, le=1.0)
    ora: float = Field(default=0.0, ge=0.0, le=1.0)
    calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    tokens: int = Field(ge=0)
    cost_per_pass_usd: float = Field(ge=0.0)
    total_cost_usd: float = Field(ge=0.0)
    per_class: dict[str, ClassScorecard] = Field(default_factory=dict)


class VariantDelta(StrictModel):
    """Difference between a variant and the baseline variant."""

    variant: str = Field(min_length=1)
    pass_at_1_delta: float
    mean_total_score_delta: float
    ambiguity_recall_delta: float
    ask_precision_delta: float
    tkqr_delta: float = 0.0
    ora_delta: float = 0.0
    over_asking_penalty_delta: float
    leakage_rate_delta: float
    avg_questions_delta: float
    avg_rounds_delta: float
    calls_delta: int
    tokens_delta: int
    cost_per_pass_usd_delta: float


class SuiteComparison(StrictModel):
    """Baseline-relative comparison across variants."""

    baseline: str = Field(min_length=1)
    deltas: dict[str, VariantDelta] = Field(default_factory=dict)


class RunMeta(StrictModel):
    """Provenance for one eval run, kept free of local host paths."""

    mode: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    variants: list[str] = Field(default_factory=list)
    case_ids: list[str] = Field(default_factory=list)
    trials: int = Field(default=1, ge=1)
    provider: str = ""
    model: str | None = None
    judge: bool = False
    generated_at: str = ""
    input_price_per_1m: float = Field(default=0.0, ge=0.0)
    output_price_per_1m: float = Field(default=0.0, ge=0.0)


class SuiteReport(StrictModel):
    """Machine-readable scorecard root object."""

    meta: RunMeta
    variants: dict[str, VariantScorecard] = Field(default_factory=dict)
    comparison: SuiteComparison


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

_STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "api",
        "code",
        "data",
        "default",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "system",
        "test",
        "tests",
        "the",
        "to",
        "with",
        "系统",
        "功能",
        "实现",
        "支持",
        "文件",
        "模块",
        "服务",
        "默认",
        "项目",
    }
)

_ASCII_TERM_RE = re.compile(r"[a-z0-9][a-z0-9 _./:-]*")


def _is_informative_fallback_term(term: str) -> bool:
    """Filter noisy fallback keywords when a slot declares no explicit terms."""
    cleaned = term.strip().casefold()
    return not (len(cleaned) < 2 or cleaned in _STOP_TERMS or cleaned.isdigit())


def _contains_literal(text: str, term: str) -> bool:
    """Match ASCII terms on token boundaries and CJK terms as substrings."""
    if _ASCII_TERM_RE.fullmatch(term):
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        return re.search(pattern, text) is not None
    return term in text


class LeakGuard:
    """Heuristic leak checks for oracle answers and goal specs."""

    @staticmethod
    def check_text_leakage(text: str, unrevealed_slots: Sequence[HiddenSlot]) -> list[str]:
        """Return forbidden terms from unrevealed slots that appear in ``text``."""
        lowered = text.casefold()
        hits: list[str] = []
        seen: set[str] = set()
        for slot in unrevealed_slots:
            if slot.forbidden_leak_terms:
                terms = slot.forbidden_leak_terms
                fallback = False
            else:
                terms = [*slot.keywords, *slot.aliases]
                fallback = True
            for term in terms:
                cleaned = term.strip().casefold()
                if not cleaned:
                    continue
                if fallback and not _is_informative_fallback_term(cleaned):
                    continue
                if cleaned in seen:
                    continue
                if _contains_literal(lowered, cleaned):
                    seen.add(cleaned)
                    hits.append(term)
        return hits

    @staticmethod
    def redact_terms(text: str, terms: Sequence[str]) -> str:
        """Case-insensitively replace leaked terms with a redaction marker."""
        redacted = text
        for term in terms:
            cleaned = term.strip()
            if not cleaned:
                continue
            if _ASCII_TERM_RE.fullmatch(cleaned.casefold()):
                pattern = rf"(?<![a-z0-9_]){re.escape(cleaned)}(?![a-z0-9_])"
            else:
                pattern = re.escape(cleaned)
            redacted = re.sub(pattern, "[redacted]", redacted, flags=re.IGNORECASE)
        return redacted


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Load a JSONL benchmark file, failing with a precise line number."""
    cases: list[EvalCase] = []
    seen_ids: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name} line {lineno}: invalid JSON ({exc.msg})"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path.name} line {lineno}: expected a JSON object")
            try:
                case = EvalCase.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"{path.name} line {lineno}: invalid EvalCase ({exc})") from exc
            if case.case_id in seen_ids:
                first_line = seen_ids[case.case_id]
                raise ValueError(
                    f"{path.name} line {lineno}: duplicate case_id '{case.case_id}' "
                    f"(first seen at line {first_line})"
                )
            seen_ids[case.case_id] = lineno
            cases.append(case)
    if not cases:
        raise ValueError(f"{path.name}: dataset contains no cases")
    return cases
