# pstack: Evidence-First Exploration

- Inspect the real codebase read-only (batch file reads efficiently) and ground every claim in exact `path:line` facts.
- Attack the premise when symptoms point to a deeper invariant violation rather than patching at the surface.
- Record solvable defaults, tie-breakers, and environment/runtime notes in `assumptions` (never in `unresolved_questions`).
- Keep `unresolved_questions` as `[]` unless execution is genuinely blocked on an unresolvable product/intent decision or irreversible external action—any item in `unresolved_questions` halts the workflow for human input.
- List the exact candidate files and any clean repo-relative `repro_command` (e.g., `python -m unittest test_foo.py`, `pytest -q`; no `cd`, `&&`, `python -c`, or host paths).
