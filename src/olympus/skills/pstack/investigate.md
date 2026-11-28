# pstack: Read-Only Root-Cause Forensics

- Operate strictly read-only (`write_scope=[]`). Never edit, create, or delete any workspace file.
- Trace the exact causal mechanism from symptom to root cause, citing concrete `path:line` evidence for every claim.
- Distinguish verified observations from hypotheses; list any remaining unknowns explicitly in `unresolved_questions`.
