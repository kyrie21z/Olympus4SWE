"""CLI helper to fetch/convert ClarifyCodeBench and Ambig-SWE datasets into EvalCase JSONL."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from tests.evals.benchmarks.ambig_swe import import_ambig_swe_files
from tests.evals.benchmarks.clarifycodebench import import_clarifycodebench_file

CLARIFYCODEBENCH_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/fangz-cs/ClarifyCodeBench/main/data/ClarifyCodeBench.jsonl"
)
AMBIG_SWE_UNDER_URL = (
    "https://raw.githubusercontent.com/sani903/InteractiveSWEAgents/main/data/underspecified.csv"
)
AMBIG_SWE_FULL_URL = (
    "https://raw.githubusercontent.com/sani903/InteractiveSWEAgents/main/data/fully-specified.csv"
)


def _download(url: str, dest: Path) -> Path:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        dest.write_bytes(resp.read())
    return dest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.evals.benchmarks.import_dataset",
        description="Convert ClarifyCodeBench or Ambig-SWE raw datasets into EvalCase JSONL.",
    )
    parser.add_argument(
        "--benchmark",
        choices=("clarifycodebench", "ambig-swe"),
        required=True,
    )
    parser.add_argument("--source", type=Path, default=None, help="ClarifyCodeBench JSONL path.")
    parser.add_argument("--underspecified", type=Path, default=None, help="Ambig-SWE under CSV.")
    parser.add_argument("--fully-specified", type=Path, default=None, help="Ambig-SWE full CSV.")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download upstream public files from GitHub when local paths are omitted.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output EvalCase JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max cases to export.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.benchmark == "clarifycodebench":
        with tempfile.TemporaryDirectory(prefix="olympus-ccb-") as tmp:
            if args.source is not None:
                source_path = args.source
            elif args.fetch:
                source_path = _download(CLARIFYCODEBENCH_UPSTREAM_URL, Path(tmp) / "ccb.jsonl")
            else:
                print("error: provide --source or --fetch", file=sys.stderr)
                return 2
            cases = import_clarifycodebench_file(
                source_path,
                args.output,
                limit=args.limit,
                seed=args.seed,
            )
    else:
        with tempfile.TemporaryDirectory(prefix="olympus-ambig-") as tmp:
            tmp_dir = Path(tmp)
            if args.underspecified is not None and args.fully_specified is not None:
                under_path = args.underspecified
                full_path = args.fully_specified
            elif args.fetch:
                under_path = _download(AMBIG_SWE_UNDER_URL, tmp_dir / "underspecified.csv")
                full_path = _download(AMBIG_SWE_FULL_URL, tmp_dir / "fully-specified.csv")
            else:
                print(
                    "error: provide both --underspecified and --fully-specified, or --fetch",
                    file=sys.stderr,
                )
                return 2
            cases = import_ambig_swe_files(
                under_path,
                full_path,
                args.output,
                limit=args.limit,
                seed=args.seed,
            )

    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "cases_written": len(cases),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
