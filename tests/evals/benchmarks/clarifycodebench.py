"""Adapter and metrics for ClarifyCodeBench (fangz-cs/ClarifyCodeBench, MIT).

Ports the canonical Turn-discounted Key Question Rate (TKQR) and Optimal Round
Adherence (ORA) definitions from ClarifyCodeBench (arXiv:2607.00711) and
converts upstream ``ClarifyCodeBench.jsonl`` tasks into ``EvalCase`` fixtures.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tests.evals.schemas import (
    BenchmarkClass,
    EvalCase,
    ExpectedCriteria,
    HiddenSlot,
    RepoFileFixture,
)

_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "allowed",
        "and",
        "any",
        "are",
        "can",
        "contain",
        "could",
        "does",
        "each",
        "else",
        "exact",
        "exactly",
        "for",
        "from",
        "have",
        "how",
        "into",
        "kinds",
        "mean",
        "must",
        "not",
        "only",
        "other",
        "otherwise",
        "printed",
        "refer",
        "refers",
        "result",
        "should",
        "some",
        "something",
        "such",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "under",
        "valid",
        "values",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "will",
        "with",
        "would",
    }
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


# ---------------------------------------------------------------------------
# Canonical ClarifyCodeBench metrics (MIT License)
# ---------------------------------------------------------------------------


def calculate_tkqr(hit_sequence: Sequence[int], total_key_questions: int) -> float:
    """Turn-discounted Key Question Rate for a single interactive task.

    ``hit_sequence[i]`` is ``1`` when the ``i``-th asked question newly uncovers
    a key ambiguity and ``0`` otherwise. Returns normalized DCG in ``[0, 1]``.
    """
    n = len(hit_sequence)
    if n == 0 or total_key_questions <= 0:
        return 0.0

    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hit_sequence))
    ideal_len = min(n, total_key_questions)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
    return dcg / idcg if idcg > 0.0 else 0.0


def ora_sigma(key_question_count: int) -> float:
    """Gaussian width parameter calibrated so ORA = 0.5 at |n - K| = 0.5*K."""
    if key_question_count <= 0:
        return 0.3
    return 0.425 * key_question_count


def calculate_ora(
    asked_count: int,
    key_question_count: int,
    sigma: float | None = None,
) -> float:
    """Optimal Round Adherence in ``(0, 1]`` peaking at ``asked_count == key_question_count``."""
    resolved_sigma = ora_sigma(key_question_count) if sigma is None else sigma
    if resolved_sigma <= 0:
        raise ValueError("sigma must be positive")
    diff = asked_count - key_question_count
    return math.exp(-(diff * diff) / (2.0 * resolved_sigma * resolved_sigma))


# ---------------------------------------------------------------------------
# Upstream row conversion & stratified selection
# ---------------------------------------------------------------------------


def _extract_question_keywords(question: str, *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for match in _WORD_RE.finditer(question):
        word = match.group(0).casefold()
        if word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        keywords.append(match.group(0))
        if len(keywords) >= limit:
            break
    if not keywords:
        fallback = question.strip()[:48]
        if fallback:
            keywords.append(fallback)
    return keywords


def _clean_tags(raw_keywords: Any) -> list[str]:
    if not isinstance(raw_keywords, Sequence) or isinstance(raw_keywords, (str, bytes)):
        return []
    return [str(tag).strip() for tag in raw_keywords if str(tag).strip()]


def convert_clarifycodebench_row(row: Mapping[str, Any]) -> EvalCase:
    """Convert one upstream ClarifyCodeBench JSON object into an ``EvalCase``."""
    task_id = str(row.get("task_id", "")).strip()
    question_id = str(row.get("question_id", "")).strip()
    modified_content = str(row.get("modified_content", "")).strip()
    raw_qa_pairs = row.get("qa_pairs")
    if not task_id or not question_id or not modified_content:
        raise ValueError("ClarifyCodeBench row requires non-empty task_id, question_id, content")
    if not isinstance(raw_qa_pairs, Sequence) or isinstance(raw_qa_pairs, (str, bytes)):
        raise ValueError(f"{task_id}: qa_pairs must be a sequence")

    slots: list[HiddenSlot] = []
    for idx, entry in enumerate(raw_qa_pairs, start=1):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{task_id}: qa_pairs[{idx - 1}] must be an object")
        question = str(entry.get("question", "")).strip()
        answer = str(entry.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError(f"{task_id}: qa_pairs[{idx - 1}] missing question/answer")
        slots.append(
            HiddenSlot(
                slot_id=f"{task_id}-S{idx}",
                topic=question,
                keywords=_extract_question_keywords(question),
                aliases=[],
                answer=answer,
                forbidden_leak_terms=[],
                blocking=True,
            )
        )

    if not slots:
        raise ValueError(f"{task_id}: expected at least 1 qa_pair")

    tags = ["clarifycodebench", *_clean_tags(row.get("keywords"))]
    raw_goal = (
        f"Clarify requirements and produce a verifiable specification for standalone task "
        f"`{question_id}` before writing code. Read `README.md` only if helpful.\n\n"
        f"{modified_content}"
    )
    readme = RepoFileFixture(
        path="README.md",
        content=(
            f"# ClarifyCodeBench Task `{question_id}`\n\n"
            "Self-contained specification alignment benchmark fixture (MIT).\n"
        ),
    )
    return EvalCase(
        case_id=f"CCB-{task_id}",
        benchmark_class=BenchmarkClass.B,
        title=f"ClarifyCodeBench {question_id} ({task_id})",
        raw_goal=raw_goal,
        initial_constraints=[
            "Identify missing specification details before freezing implementation.",
            "Ask concise targeted questions instead of guessing unstated requirements.",
        ],
        hidden_slots=slots,
        repo_files=[readme],
        expected=ExpectedCriteria(
            expect_refusal_or_pause=False,
            min_ambiguity_recall=1.0,
            max_questions=max(3, len(slots) + 2),
            max_clarify_rounds=max(2, len(slots)),
            require_repo_conflict_check=False,
            require_confirmable_at_end=True,
            require_downstream_plan_valid=False,
            leak_check=False,
        ),
        tags=tags,
    )


def load_clarifycodebench_rows(path: Path) -> list[dict[str, Any]]:
    """Load raw JSONL objects from an upstream ClarifyCodeBench file."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path.name} line {lineno}: expected JSON object")
            rows.append(payload)
    if not rows:
        raise ValueError(f"{path.name}: no rows found")
    return rows


def select_stratified_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministically sample ``limit`` rows stratified across K and primary keyword."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    items = [dict(row) for row in rows]
    if limit >= len(items):
        return items

    buckets: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in items:
        qa = row.get("qa_pairs")
        k = len(qa) if isinstance(qa, Sequence) else 0
        kws = _clean_tags(row.get("keywords"))
        primary = kws[0] if kws else "Uncategorized"
        buckets.setdefault((k, primary), []).append(row)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    bucket_keys = sorted(buckets)
    idx = 0
    while len(selected) < limit and any(buckets.values()):
        key = bucket_keys[idx % len(bucket_keys)]
        bucket = buckets[key]
        if bucket:
            selected.append(bucket.pop(0))
        idx += 1

    order_index = {str(row.get("task_id")): pos for pos, row in enumerate(items)}
    selected.sort(key=lambda row: order_index.get(str(row.get("task_id")), 0))
    return selected


def import_clarifycodebench_file(
    source: Path,
    dest: Path,
    *,
    limit: int | None = None,
    seed: int = 42,
) -> list[EvalCase]:
    """Convert upstream ``ClarifyCodeBench.jsonl`` into Olympus ``EvalCase`` JSONL."""
    rows = load_clarifycodebench_rows(source)
    chosen = select_stratified_rows(rows, limit=limit, seed=seed) if limit is not None else rows
    cases = [convert_clarifycodebench_row(row) for row in chosen]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(case.model_dump_json() + "\n")
    return cases
