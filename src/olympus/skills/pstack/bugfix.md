# pstack: Root-Cause Bugfix Discipline

- Reproduce the defect deterministically on `base_ref` before proposing or writing any fix. Never patch speculatively.
- Trace the causal chain to the invariant violation and fix the root cause—never mask symptoms with defensive null-guards, broad try/except blocks, or weakened assertions.
- Lock the defect with a failing test before the fix and confirm the exact same test surface turns green post-fix.
