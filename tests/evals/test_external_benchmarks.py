"""Unit tests for ClarifyCodeBench/Ambig-SWE adapters, TKQR/ORA, and semantic oracle credit."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from olympus.core.contracts import GoalSpec
from tests.evals.benchmarks.ambig_swe import (
    convert_ambig_swe_pair,
    import_ambig_swe_files,
    load_ambig_swe_pairs,
)
from tests.evals.benchmarks.clarifycodebench import (
    calculate_ora,
    calculate_tkqr,
    convert_clarifycodebench_row,
    import_clarifycodebench_file,
    select_stratified_rows,
)
from tests.evals.runner import EvalRunner
from tests.evals.schemas import (
    BenchmarkClass,
    EvalCase,
    ExpectedCriteria,
    HiddenSlot,
    RepoFileFixture,
    load_eval_cases,
)
from tests.evals.scorers import CaseEvidence, evaluate_case_outcome

CCB_30_PATH = Path(__file__).resolve().parent / "datasets" / "clarifycodebench_30.jsonl"


def test_tkqr_and_ora_canonical_math() -> None:
    assert calculate_tkqr([], 2) == 0.0
    assert calculate_tkqr([1], 0) == 0.0
    assert calculate_tkqr([1, 1], 2) == pytest.approx(1.0)
    assert calculate_tkqr([1, 0, 1], 2) == pytest.approx(
        (1.0 / math.log2(2) + 1.0 / math.log2(4))
        / (1.0 / math.log2(2) + 1.0 / math.log2(3))
    )
    assert calculate_ora(2, 2) == pytest.approx(1.0)
    # 50% relative deviation yields 0.50 up to the 0.425 approximation constant.
    assert calculate_ora(3, 2) == pytest.approx(math.exp(-1.0 / (2.0 * (0.85**2))))
    assert calculate_ora(0, 0) == pytest.approx(1.0)
    assert calculate_ora(2, 0) < 0.01
    with pytest.raises(ValueError):
        calculate_ora(1, 1, sigma=0.0)


def test_clarifycodebench_converter_and_stratification(tmp_path: Path) -> None:
    rows = [
        {
            "task_id": f"t_{i}",
            "question_id": f"q_{i}",
            "keywords": ["Terminology" if i % 2 == 0 else "Output Format"],
            "modified_content": f"Problem statement {i}",
            "qa_pairs": [
                {"question": f"Clarify constraint A for {i}?", "answer": "Answer A"}
            ]
            * (1 + (i % 3)),
        }
        for i in range(12)
    ]
    sample = select_stratified_rows(rows, limit=6, seed=7)
    assert len(sample) == 6
    assert {len(item["qa_pairs"]) for item in sample} == {1, 2, 3}

    case = convert_clarifycodebench_row(rows[1])
    assert case.case_id == "CCB-t_1"
    assert case.benchmark_class == BenchmarkClass.B
    assert len(case.hidden_slots) == 2
    assert case.expected.leak_check is False
    assert "Output Format" in case.tags

    src = tmp_path / "ccb.jsonl"
    dst = tmp_path / "out.jsonl"
    import json

    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    imported = import_clarifycodebench_file(src, dst, limit=6, seed=7)
    assert len(imported) == 6
    assert len(load_eval_cases(dst)) == 6


def test_committed_clarifycodebench_30_dataset_is_valid_and_runnable_mock(
    tmp_path: Path,
) -> None:
    cases = load_eval_cases(CCB_30_PATH)
    assert len(cases) == 30
    k_values = {len(case.hidden_slots) for case in cases}
    assert k_values == {1, 2, 3}

    runner = EvalRunner(
        dataset_path=CCB_30_PATH,
        output_dir=tmp_path / "ccb-mock-run",
        mode="mock",
        variants=["baseline"],
    )
    report = runner.run()
    baseline = report.variants["baseline"]
    assert baseline.cases == 30
    assert baseline.pass_at_1 == pytest.approx(1.0)
    assert baseline.tkqr == pytest.approx(1.0)
    assert baseline.ora == pytest.approx(1.0)
    assert baseline.leakage_rate == pytest.approx(0.0)


def test_ambig_swe_csv_importer(tmp_path: Path) -> None:
    under = tmp_path / "underspecified.csv"
    full = tmp_path / "fully-specified.csv"
    under.write_text(
        'instance_id,problem_statement\nrepo__pkg-123,"Fix broken error message in loader."\n',
        encoding="utf-8",
    )
    full.write_text(
        'instance_id,problem_statement\n'
        'repo__pkg-123,"Fix broken error message in loader. '
        "When column 'ts' is missing, raise KeyError naming 'ts' explicitly. "
        'Preserve backwards compatibility for JSON inputs."\n',
        encoding="utf-8",
    )
    pairs = load_ambig_swe_pairs(under, full)
    assert len(pairs) == 1
    case = convert_ambig_swe_pair(pairs[0])
    assert case.case_id == "AMBIG-repo__pkg-123"
    assert "repo/pkg" in case.tags
    assert len(case.hidden_slots) == 2

    out = tmp_path / "ambig.jsonl"
    cases = import_ambig_swe_files(under, full, out)
    assert len(cases) == 1
    assert load_eval_cases(out)[0].case_id == "AMBIG-repo__pkg-123"


def test_evaluate_credits_semantic_oracle_matches_when_keywords_miss(
    tmp_path: Path,
) -> None:
    case = EvalCase(
        case_id="CCB-semantic-1",
        benchmark_class=BenchmarkClass.B,
        title="Paraphrase semantic oracle credit",
        raw_goal="Print whether array satisfies property.",
        hidden_slots=[
            HiddenSlot(
                slot_id="S1",
                topic="Should output be printed as Yes/No or true/false?",
                keywords=["Yes/No", "true/false"],
                answer="Print Yes or No.",
                blocking=True,
            )
        ],
        repo_files=[RepoFileFixture(path="README.md", content="# Task\n")],
        expected=ExpectedCriteria(
            expect_refusal_or_pause=False,
            min_ambiguity_recall=1.0,
            max_questions=3,
            max_clarify_rounds=2,
            require_confirmable_at_end=True,
            leak_check=False,
        ),
    )
    spec = GoalSpec.model_validate(
        {
            "goal": "Print whether array satisfies property.",
            "requirements": [
                {
                    "id": "R1",
                    "description": (
                        "Handle valid inputs cleanly; raise ValueError on malformed input."
                    ),
                    "acceptance_criteria": [
                        "`python3 -m compileall -q .` exit code == 0",
                        "Invalid input exits cleanly with error message",
                    ],
                },
                {
                    "id": "R2",
                    "description": "Reject boundary violations and report failure cleanly.",
                    "acceptance_criteria": ["Boundary checks pass assertions"],
                },
            ],
            "out_of_scope": ["Extra CLI flags"],
            "open_ambiguities": [
                {
                    "id": "AMB-1",
                    "question": "Which exact capitalization/tokens should stdout emit?",
                    "impact": "Affects CLI stdout verification",
                    "options": [],
                    "blocking": True,
                    "answer": "Print Yes or No.",
                }
            ],
        }
    )

    # Without oracle semantic credit, keyword matcher returns 0 recall on the paraphrase.
    keyword_only = CaseEvidence(
        case=case,
        initial_spec=spec,
        final_spec=spec,
        refusal_confirm_rejected=False,
        final_confirmable=True,
        asked_questions=("Which exact capitalization/tokens should stdout emit?",),
        clarify_rounds=1,
        repeated_question_rate=0.0,
        revealed_slot_ids=frozenset({"S1"}),
        oracle_leakage_detected=False,
        oracle_leak_terms=(),
        plan=None,
        workspace_root=tmp_path,
        question_hits=(),
        oracle_evaluated=False,
    )
    passed_kw, _, metrics_kw = evaluate_case_outcome(keyword_only)
    assert passed_kw is False
    assert metrics_kw.ambiguity_recall == 0.0

    # With oracle_evaluated=True, LLMOracle's semantic slot reveal is credited cleanly.
    semantic_credited = CaseEvidence(
        case=case,
        initial_spec=spec,
        final_spec=spec,
        refusal_confirm_rejected=False,
        final_confirmable=True,
        asked_questions=("Which exact capitalization/tokens should stdout emit?",),
        clarify_rounds=1,
        repeated_question_rate=0.0,
        revealed_slot_ids=frozenset({"S1"}),
        oracle_leakage_detected=False,
        oracle_leak_terms=(),
        plan=None,
        workspace_root=tmp_path,
        question_hits=(1,),
        oracle_evaluated=True,
    )
    passed_sem, _, metrics_sem = evaluate_case_outcome(semantic_credited)
    assert passed_sem is True
    assert metrics_sem.ambiguity_recall == pytest.approx(1.0)
    assert metrics_sem.ask_precision == pytest.approx(1.0)
    assert metrics_sem.tkqr == pytest.approx(1.0)
    assert metrics_sem.ora == pytest.approx(1.0)
