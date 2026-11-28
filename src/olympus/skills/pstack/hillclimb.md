# pstack: Frozen-Harness Metric Hill-Climbing

- Formulate ONE grounded, falsifiable optimization hypothesis per iteration and state it clearly as `HYPOTHESIS: <concise hypothesis>`.
- Edit strictly inside `write_scope`; NEVER modify, touch, or work around any file in `frozen_harness_paths` (SHA-256 anti-tamper guard rejects and reverts any harness edit immediately).
- Preserve functional correctness alongside metric improvement; do not hardcode benchmark inputs or cache outputs across runs.
- Review prior kept and reverted iterations so each new hypothesis explores a genuinely distinct optimization lever (e.g., algorithmic complexity reduction, better data structures, eliminating redundant work).
