"""Aggregation and artifact writing for Stage 1 eval runs.

Outputs are intentionally path-free: ``run_meta.json`` records only the dataset
file name and run parameters so committed or shared artifacts never leak a
local home directory.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from tests.evals.schemas import (
    CaseRunRecord,
    ClassScorecard,
    RunMeta,
    SuiteComparison,
    SuiteReport,
    VariantDelta,
    VariantScorecard,
)


def read_records(path: Path) -> list[CaseRunRecord]:
    """Read an append-only ``records.jsonl`` file with line-accurate errors."""
    if not path.exists():
        return []
    records: list[CaseRunRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(CaseRunRecord.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"{path.name} line {lineno}: invalid record ({exc})") from exc
    return records


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pass_all_rate(records: Sequence[CaseRunRecord], trials: int) -> float | None:
    if trials <= 1:
        return None
    by_case: dict[str, list[CaseRunRecord]] = {}
    for record in records:
        by_case.setdefault(record.case_id, []).append(record)
    complete: list[list[CaseRunRecord]] = []
    for group in by_case.values():
        seen_trials = {record.trial for record in group}
        if seen_trials >= set(range(1, trials + 1)):
            complete.append(group)
    if not complete:
        return None
    all_passed = sum(1 for group in complete if all(record.passed for record in group))
    return all_passed / len(complete)


def _class_scorecard(records: Sequence[CaseRunRecord], trials: int) -> ClassScorecard:
    passed = sum(1 for record in records if record.passed)
    case_count = len({record.case_id for record in records})
    return ClassScorecard(
        cases=case_count,
        records=len(records),
        passed=passed,
        pass_at_1=passed / len(records) if records else 0.0,
        pass_all=_pass_all_rate(records, trials),
        mean_total_score=_mean([record.total_score for record in records]),
        ambiguity_recall=_mean([record.metrics.ambiguity_recall for record in records]),
        ask_precision=_mean([record.metrics.ask_precision for record in records]),
        over_asking_penalty=_mean([record.metrics.over_asking_penalty for record in records]),
        leakage_rate=_mean(
            [1.0 if record.metrics.leakage_detected else 0.0 for record in records]
        ),
        avg_questions=_mean([float(record.metrics.questions_asked) for record in records]),
        avg_rounds=_mean([float(record.metrics.clarify_rounds) for record in records]),
    )


def summarize_variant(
    variant: str,
    records: Sequence[CaseRunRecord],
    *,
    trials: int,
) -> VariantScorecard:
    """Aggregate one variant's records into a scorecard."""
    subset = [record for record in records if record.variant == variant]
    passed = sum(1 for record in subset if record.passed)
    input_tokens = sum(record.input_tokens for record in subset)
    output_tokens = sum(record.output_tokens for record in subset)
    total_cost = sum(record.estimated_cost_usd for record in subset)

    by_class: dict[str, list[CaseRunRecord]] = {}
    for record in subset:
        by_class.setdefault(record.benchmark_class.value, []).append(record)
    per_class = {
        class_name: _class_scorecard(group, trials)
        for class_name, group in sorted(by_class.items())
    }

    return VariantScorecard(
        variant=variant,
        cases=len({record.case_id for record in subset}),
        records=len(subset),
        passed=passed,
        pass_at_1=passed / len(subset) if subset else 0.0,
        pass_all=_pass_all_rate(subset, trials),
        mean_total_score=_mean([record.total_score for record in subset]),
        ambiguity_recall=_mean([record.metrics.ambiguity_recall for record in subset]),
        ask_precision=_mean([record.metrics.ask_precision for record in subset]),
        over_asking_penalty=_mean([record.metrics.over_asking_penalty for record in subset]),
        leakage_rate=_mean(
            [1.0 if record.metrics.leakage_detected else 0.0 for record in subset]
        ),
        avg_questions=_mean([float(record.metrics.questions_asked) for record in subset]),
        avg_rounds=_mean([float(record.metrics.clarify_rounds) for record in subset]),
        calls=sum(record.llm_calls for record in subset),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokens=input_tokens + output_tokens,
        cost_per_pass_usd=(total_cost / passed) if passed else 0.0,
        total_cost_usd=total_cost,
        per_class=per_class,
    )


def _delta(reference: VariantScorecard, candidate: VariantScorecard) -> VariantDelta:
    return VariantDelta(
        variant=candidate.variant,
        pass_at_1_delta=candidate.pass_at_1 - reference.pass_at_1,
        mean_total_score_delta=candidate.mean_total_score - reference.mean_total_score,
        ambiguity_recall_delta=candidate.ambiguity_recall - reference.ambiguity_recall,
        ask_precision_delta=candidate.ask_precision - reference.ask_precision,
        over_asking_penalty_delta=(
            candidate.over_asking_penalty - reference.over_asking_penalty
        ),
        leakage_rate_delta=candidate.leakage_rate - reference.leakage_rate,
        avg_questions_delta=candidate.avg_questions - reference.avg_questions,
        avg_rounds_delta=candidate.avg_rounds - reference.avg_rounds,
        calls_delta=candidate.calls - reference.calls,
        tokens_delta=candidate.tokens - reference.tokens,
        cost_per_pass_usd_delta=(
            candidate.cost_per_pass_usd - reference.cost_per_pass_usd
        ),
    )


def build_report(
    *,
    meta: RunMeta,
    records: Sequence[CaseRunRecord],
) -> SuiteReport:
    """Build the machine-readable scorecard root object."""
    variants = {
        variant: summarize_variant(variant, records, trials=meta.trials)
        for variant in meta.variants
    }
    baseline_name = "baseline" if "baseline" in variants else next(iter(variants), "baseline")
    reference = variants.get(baseline_name)
    deltas: dict[str, VariantDelta] = {}
    if reference is not None:
        for name, scorecard in variants.items():
            if name == baseline_name:
                continue
            deltas[name] = _delta(reference, scorecard)
    return SuiteReport(
        meta=meta,
        variants=variants,
        comparison=SuiteComparison(baseline=baseline_name, deltas=deltas),
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _variant_table(report: SuiteReport) -> list[str]:
    header = (
        "| Variant | Cases | Records | pass@1 | pass_all | mean_total_score | "
        "ambiguity_recall | ask_precision | over_asking_penalty | leakage_rate | "
        "avg_questions | avg_rounds | calls | tokens | cost_per_pass_usd |"
    )
    separator = "|" + "---|" * 15
    rows = [
        header,
        separator,
    ]
    for scorecard in report.variants.values():
        rows.append(
            f"| {scorecard.variant} | {scorecard.cases} | {scorecard.records} | "
            f"{_fmt(scorecard.pass_at_1)} | {_fmt(scorecard.pass_all)} | "
            f"{_fmt(scorecard.mean_total_score)} | {_fmt(scorecard.ambiguity_recall)} | "
            f"{_fmt(scorecard.ask_precision)} | {_fmt(scorecard.over_asking_penalty)} | "
            f"{_fmt(scorecard.leakage_rate)} | {scorecard.avg_questions:.2f} | "
            f"{scorecard.avg_rounds:.2f} | {scorecard.calls} | {scorecard.tokens} | "
            f"${scorecard.cost_per_pass_usd:.4f} |"
        )
    return rows


def _per_class_table(report: SuiteReport) -> list[str]:
    header = (
        "| Variant | Class | Cases | Records | passed | pass@1 | pass_all | "
        "mean_total_score | ambiguity_recall | ask_precision | over_asking_penalty | "
        "leakage_rate | avg_questions | avg_rounds |"
    )
    separator = "|" + "---|" * 14
    rows = [header, separator]
    for scorecard in report.variants.values():
        for class_name, class_score in scorecard.per_class.items():
            rows.append(
                f"| {scorecard.variant} | {class_name} | {class_score.cases} | "
                f"{class_score.records} | {class_score.passed} | "
                f"{_fmt(class_score.pass_at_1)} | {_fmt(class_score.pass_all)} | "
                f"{_fmt(class_score.mean_total_score)} | "
                f"{_fmt(class_score.ambiguity_recall)} | {_fmt(class_score.ask_precision)} | "
                f"{_fmt(class_score.over_asking_penalty)} | {_fmt(class_score.leakage_rate)} | "
                f"{class_score.avg_questions:.2f} | {class_score.avg_rounds:.2f} |"
            )
    return rows


def _delta_table(report: SuiteReport) -> list[str]:
    if not report.comparison.deltas:
        return ["No comparison variants were executed."]
    header = (
        "| Variant | baseline | Δpass@1 | Δmean_total_score | Δambiguity_recall | "
        "Δask_precision | Δover_asking_penalty | Δleakage_rate | Δcalls | Δtokens | "
        "Δcost_per_pass_usd |"
    )
    separator = "|" + "---|" * 11
    rows = [header, separator]
    for delta in report.comparison.deltas.values():
        rows.append(
            f"| {delta.variant} | {report.comparison.baseline} | "
            f"{delta.pass_at_1_delta:+.3f} | {delta.mean_total_score_delta:+.3f} | "
            f"{delta.ambiguity_recall_delta:+.3f} | {delta.ask_precision_delta:+.3f} | "
            f"{delta.over_asking_penalty_delta:+.3f} | {delta.leakage_rate_delta:+.3f} | "
            f"{delta.calls_delta:+d} | {delta.tokens_delta:+d} | "
            f"${delta.cost_per_pass_usd_delta:+.4f} |"
        )
    return rows


def render_markdown(report: SuiteReport) -> str:
    """Render a clean, review-friendly Markdown scorecard."""
    meta = report.meta
    lines = [
        "# Olympus Stage 1 Eval Scorecard",
        "",
        f"- Mode: `{meta.mode}`",
        f"- Dataset: `{meta.dataset}`",
        f"- Cases: {len(meta.case_ids)}",
        f"- Variants: {', '.join(meta.variants)}",
        f"- Trials: {meta.trials}",
        f"- Judge: {'on' if meta.judge else 'off'}",
        f"- Generated: {meta.generated_at}",
        "",
        "## Variant Summary",
        "",
        *_variant_table(report),
        "",
        "## Per-Class Breakdown",
        "",
        *_per_class_table(report),
        "",
        "## Delta vs Baseline",
        "",
        *_delta_table(report),
        "",
        "## Notes",
        "",
        "- `pass@1` is the mean pass rate across all trials.",
        "- `passed` counts passing records; `cases` counts distinct case ids.",
        "- `pass_all` is reported only when `trials > 1`.",
        "- `cost_per_pass_usd` is 0 when the variant has no passing records.",
        "",
    ]
    return "\n".join(lines)


def generate_reports(
    *,
    output_dir: Path,
    meta: RunMeta,
    records: Sequence[CaseRunRecord],
) -> SuiteReport:
    """Write ``run_meta.json``, ``results.json``, ``scorecard.json`` and ``scorecard.md``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "results.json").write_text(
        json.dumps(
            [record.model_dump(mode="json") for record in records],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_report(meta=meta, records=records)
    (output_dir / "scorecard.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "scorecard.md").write_text(render_markdown(report) + "\n", encoding="utf-8")
    return report
