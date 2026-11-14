# External Benchmark Dataset Attribution

- **`clarifycodebench_30.jsonl`**:
  - Source: [ClarifyCodeBench](https://github.com/fangz-cs/ClarifyCodeBench) (`data/ClarifyCodeBench.jsonl`), *arXiv:2607.00711* (`MIT License`).
  - Selection: Stratified sample of 30 tasks across `K in {1, 2, 3}` key questions and ambiguity taxonomies (`seed=42`) generated via `python -m tests.evals.benchmarks.import_dataset --benchmark clarifycodebench --fetch --limit 30 --seed 42`.
  - Metrics ported under MIT License: `TKQR` (Turn-discounted Key Question Rate) and `ORA` (Optimal Round Adherence) in `tests/evals/benchmarks/clarifycodebench.py`.

- **Ambig-SWE Importer (`tests/evals/benchmarks/ambig_swe.py`)**:
  - Target upstream dataset: [InteractiveSWEAgents / Ambig-SWE](https://github.com/sani903/InteractiveSWEAgents) (`arXiv:2502.13069`, `underspecified.csv` + `fully-specified.csv`).
  - Usage: Run `python -m tests.evals.benchmarks.import_dataset --benchmark ambig-swe --fetch --limit 20 --output tests/evals/datasets/ambig_swe_20.jsonl` whenever generating repo-level SWE clarification fixtures locally.
