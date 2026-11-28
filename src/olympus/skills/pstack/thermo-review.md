# pstack: Thermo-Nuclear Dual-Lane Audit

- Correctness lane: hunt diff-scoped logic bugs, contract/signature breakage, security/path-escape flaws, and unhandled failure modes.
- Code-Judo lane: flag dramatic missed simplifications, thin redundant wrappers, file explosion, or files exceeding cohesion bounds.
- Never report unfinished speculation; ground every finding in the diff and runnable checks.
