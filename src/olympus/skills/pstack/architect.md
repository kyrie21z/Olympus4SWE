# pstack: Two-Shape Architecture Sketch

- Name the data shape first and compare at least two genuinely distinct candidate shapes before choosing one.
- Enforce one owner per state, one canonical path per task, and least caller knowledge across module boundaries.
- Subtract before adding: prefer removing indirection, thin wrappers, and legacy paths over layering new abstractions.
- Emit a tight, minimal `write_scope` covering only files that must change, plus concrete executable shell commands in `verification_plan` that verify real behavior.
