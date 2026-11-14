"""CLI entry point for the Stage 1 eval harness.

Examples::

    uv run python -m tests.evals.run_eval --dry-run
    uv run python -m tests.evals.run_eval --mock --variants baseline,critique
    uv run python -m tests.evals.run_eval --live --provider openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tests.evals.runner import EvalMode, EvalRunner
from tests.evals.schemas import BenchmarkClass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_dataset() -> Path:
    return Path(__file__).resolve().parent / "datasets" / "stage1_4class.jsonl"


def _default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return _repo_root() / "tests" / "evals" / "results" / f"run-{stamp}"


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.evals.run_eval",
        description="Run the Stage 1 four-class benchmark and write scorecards.",
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Path to a JSONL dataset.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--mock",
        action="store_true",
        help="Use scripted offline providers (default).",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Use a real provider resolved by olympus.providers.factory.",
    )
    parser.add_argument("--provider", default="auto", help="Provider id for --live mode.")
    parser.add_argument("--model", default=None, help="Override the live model name.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help="HTTP timeout in seconds per live LLM call (default: 240).",
    )
    parser.add_argument(
        "--variants",
        default="baseline,critique",
        help="Comma-separated variants (baseline, critique).",
    )
    parser.add_argument("--classes", default=None, help="Comma-separated classes (A,B,C,D).")
    parser.add_argument("--case-ids", default=None, help="Comma-separated case ids.")
    parser.add_argument("--trials", type=int, default=1, help="Trials per (case, variant).")
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help="Attach an LLM judge verdict to every converged case.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate fixtures and print call-bound estimates without calling models.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Directory containing records.jsonl; completed runs are skipped.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for records and scorecards.",
    )
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=0.0,
        help="Input token price in USD per 1M tokens (cost reporting only).",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=0.0,
        help="Output token price in USD per 1M tokens (cost reporting only).",
    )
    return parser


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _parse_classes(raw: str | None) -> list[BenchmarkClass] | None:
    if not raw:
        return None
    try:
        return [BenchmarkClass(value.upper()) for value in _split_csv(raw)]
    except ValueError as exc:
        raise ValueError(f"invalid --classes value: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode: EvalMode = "live" if args.live else "mock"
    dataset = args.dataset if args.dataset is not None else _default_dataset()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (args.resume if args.resume is not None else _default_output_dir())
    )
    variants = _split_csv(args.variants) or ["baseline", "critique"]
    case_ids = _split_csv(args.case_ids) if args.case_ids else None

    try:
        runner = EvalRunner(
            dataset_path=dataset,
            output_dir=output_dir,
            mode=mode,
            variants=variants,
            classes=_parse_classes(args.classes),
            case_ids=case_ids,
            trials=args.trials,
            with_judge=args.with_judge,
            provider=args.provider,
            model=args.model,
            request_timeout=args.timeout,
            input_price_per_1m=args.input_price_per_1m,
            output_price_per_1m=args.output_price_per_1m,
            resume_dir=args.resume,
        )
        if args.dry_run:
            summary = runner.dry_run()
            _emit(summary)
            return 0
        report = runner.run()
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _emit(
        {
            "output_dir": str(output_dir),
            "mode": mode,
            "records": sum(scorecard.records for scorecard in report.variants.values()),
            "variants": {
                name: {
                    "pass@1": scorecard.pass_at_1,
                    "mean_total_score": scorecard.mean_total_score,
                    "leakage_rate": scorecard.leakage_rate,
                    "calls": scorecard.calls,
                }
                for name, scorecard in report.variants.items()
            },
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
