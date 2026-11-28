# pstack: Diff Hygiene (Deslop & Comment Discipline)

- Review ONLY the current diff (`base..HEAD`) inside `write_scope`. Keep behavior strictly identical.
- If the diff is already concise and free of AI slop or redundant comments, finish immediately in 1 step with zero edits.
- Otherwise strip AI slop: redundant prose comments explaining obvious code, abnormal defensive try/catch blocks, `any`-casts, dead helpers, and unnecessary deep nesting.
- Keep comments only when required by license headers, public API docstrings, external tooling directives, or non-obvious external constraints.
