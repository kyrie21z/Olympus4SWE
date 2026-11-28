# pstack: Multi-Worktree Arena, Rubric Judge & Graft

- Define 3 to 6 concrete, weighted rubric criteria anchored to correctness, boundary simplicity, verification rigor, and maintainability before scoring any candidate.
- Build each candidate independently inside its isolated git worktree without peeking at sibling attempts; obey `write_scope` strictly and leave zero ephemeral files behind.
- Judge cold-start with zero builder history: score every candidate against every rubric criterion on run-verified evidence and diff quality, and never select a `DiffGate`-rejected candidate as winner.
- Graft at most 1 to 2 specific, high-leverage improvements from non-winning candidates onto the winner without regressing verification or expanding `write_scope`.
