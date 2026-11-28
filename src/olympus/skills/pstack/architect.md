# pstack: Two-Shape Architecture Sketch

- Name the data shape first and compare at least two genuinely distinct candidate shapes before choosing one.
- Enforce one owner per state, one canonical path per task, and least caller knowledge across module boundaries.
- Subtract before adding: prefer removing indirection, thin wrappers, and legacy paths over layering new abstractions.
- Emit a tight, minimal `write_scope` covering only files that must change.
- Every entry in `verification_plan` MUST be a direct read-only test/lint/typecheck command executed from the repo root with `shell=False` (e.g. `python -m unittest test_foo.py -v`, `pytest -q`, `ruff check .`, `mypy src`, `npm test`). NEVER use `cd`, `&&`, `;`, `|`, `>`, `python -c`, `grep`, mutating flags (`--fix`, `-w`), or absolute host paths.
- ONLY include tools/linters that are actually configured in the repository or explicitly requested by the user (do not invent `mypy`/`ruff`/`pytest` checks for minimal repos that only use `unittest`).
