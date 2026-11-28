# pstack: Evidence-First Exploration

- Inspect the real codebase read-only before proposing any change. Ground every claim in exact `path:line` facts.
- Attack the premise when symptoms point to a deeper invariant violation rather than patching at the surface.
- Never block on the human for solvable codebase questions. Populate `unresolved_questions` ONLY for genuinely unresolvable product/intent ambiguity or irreversible external actions.
- List the exact candidate files and any runnable reproduction command observed in the repo.
