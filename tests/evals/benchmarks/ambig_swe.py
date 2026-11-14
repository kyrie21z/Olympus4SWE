"""Adapter for Ambig-SWE / InteractiveSWEAgents (sani903/InteractiveSWEAgents).

Converts paired ``underspecified.csv`` and ``fully-specified.csv`` rows derived
from SWE-bench Verified into ``EvalCase`` JSONL fixtures without requiring the
upstream OpenHands/Docker runtime.
"""

from __future__ import annotations

import csv
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tests.evals.schemas import (
    BenchmarkClass,
    EvalCase,
    ExpectedCriteria,
    HiddenSlot,
    RepoFileFixture,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{2,}")
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "and",
        "are",
        "but",
        "can",
        "does",
        "for",
        "from",
        "have",
        "into",
        "issue",
        "not",
        "should",
        "that",
        "the",
        "this",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
)


@dataclass(frozen=True)
class AmbigSWEPair:
    """One underspecified/fully-specified issue pair keyed by SWE-bench instance_id."""

    instance_id: str
    underspecified_statement: str
    fully_specified_statement: str


def _normalize_clause(text: str) -> str:
    return " ".join(text.split()).casefold()


def _extract_clause_keywords(clause: str, *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for match in _WORD_RE.finditer(clause):
        token = match.group(0)
        lowered = token.casefold()
        if lowered in _STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    if not keywords and clause.strip():
        keywords.append(clause.strip()[:48])
    return keywords


def _extract_missing_clauses(
    underspecified: str,
    fully_specified: str,
    *,
    max_slots: int = 4,
) -> list[str]:
    """Identify sentences/paragraphs present in ``fully_specified`` but missing upfront."""
    under_norm = _normalize_clause(underspecified)
    raw_clauses = [
        " ".join(chunk.split())
        for chunk in _SENTENCE_SPLIT_RE.split(fully_specified)
        if chunk and chunk.strip()
    ]
    missing: list[str] = []
    for clause in raw_clauses:
        if len(clause) < 20:
            continue
        norm = clause.casefold()
        if norm in under_norm:
            continue
        missing.append(clause[:500])
        if len(missing) >= max_slots:
            break

    if not missing:
        fallback = " ".join(fully_specified.split())[:600]
        if fallback:
            missing.append(fallback)
    return missing


def load_ambig_swe_pairs(
    underspecified_csv: Path,
    fully_specified_csv: Path,
) -> list[AmbigSWEPair]:
    """Join Ambig-SWE CSV tables on ``instance_id`` preserving CSV order."""
    with fully_specified_csv.open("r", encoding="utf-8", newline="") as handle:
        full_map: dict[str, str] = {}
        for row in csv.DictReader(handle):
            instance_id = (row.get("instance_id") or "").strip()
            statement = (row.get("problem_statement") or "").strip()
            if instance_id and statement:
                full_map[instance_id] = statement

    pairs: list[AmbigSWEPair] = []
    with underspecified_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            instance_id = (row.get("instance_id") or "").strip()
            under = (row.get("problem_statement") or "").strip()
            full = full_map.get(instance_id, "")
            if not instance_id or not under or not full:
                continue
            pairs.append(
                AmbigSWEPair(
                    instance_id=instance_id,
                    underspecified_statement=under,
                    fully_specified_statement=full,
                )
            )
    if not pairs:
        raise ValueError("No matched instance_id pairs found across Ambig-SWE CSV files")
    return pairs


def _repo_slug(instance_id: str) -> str:
    prefix = instance_id.rsplit("-", 1)[0]
    return prefix.replace("__", "/")


def convert_ambig_swe_pair(
    pair: AmbigSWEPair,
    *,
    repo_files: Sequence[RepoFileFixture] = (),
) -> EvalCase:
    """Convert one ``AmbigSWEPair`` into a Stage 1 clarification ``EvalCase``."""
    clauses = _extract_missing_clauses(
        pair.underspecified_statement,
        pair.fully_specified_statement,
    )
    slots = [
        HiddenSlot(
            slot_id=f"{pair.instance_id}-S{idx}",
            topic=clause[:140],
            keywords=_extract_clause_keywords(clause),
            aliases=[],
            answer=clause,
            forbidden_leak_terms=[],
            blocking=True,
        )
        for idx, clause in enumerate(clauses, start=1)
    ]
    slug = _repo_slug(pair.instance_id)
    fixtures = list(repo_files) or [
        RepoFileFixture(
            path="README.md",
            content=(
                f"# Ambig-SWE Instance `{pair.instance_id}`\n\n"
                f"Upstream repository: `{slug}`.\n"
            ),
        )
    ]
    return EvalCase(
        case_id=f"AMBIG-{pair.instance_id}",
        benchmark_class=BenchmarkClass.B,
        title=f"Ambig-SWE {pair.instance_id}",
        raw_goal=pair.underspecified_statement,
        initial_constraints=[
            "Probe missing details before committing implementation assumptions."
        ],
        hidden_slots=slots,
        repo_files=fixtures,
        expected=ExpectedCriteria(
            expect_refusal_or_pause=False,
            min_ambiguity_recall=1.0,
            max_questions=max(4, len(slots) + 2),
            max_clarify_rounds=3,
            require_repo_conflict_check=False,
            require_confirmable_at_end=True,
            require_downstream_plan_valid=False,
            leak_check=False,
        ),
        tags=["ambig-swe", slug],
    )


def import_ambig_swe_files(
    underspecified_csv: Path,
    fully_specified_csv: Path,
    dest: Path,
    *,
    limit: int | None = None,
    seed: int = 42,
) -> list[EvalCase]:
    """Convert matched Ambig-SWE pairs into ``EvalCase`` JSONL."""
    pairs = load_ambig_swe_pairs(underspecified_csv, fully_specified_csv)
    if limit is not None and limit < len(pairs):
        if limit <= 0:
            raise ValueError("limit must be positive")
        rng = random.Random(seed)
        chosen_indices = sorted(rng.sample(range(len(pairs)), limit))
        pairs = [pairs[i] for i in chosen_indices]

    cases = [convert_ambig_swe_pair(pair) for pair in pairs]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(case.model_dump_json() + "\n")
    return cases
