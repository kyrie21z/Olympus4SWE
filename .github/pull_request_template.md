Closes #<issue-id>

<!-- Review standard: all PRs are evaluated against docs/REVIEW.md (automated Hydrogen review and required CI checks). -->

## Summary

<!-- Keep it short and factual. Include What changed, How tested, and Risks (required by docs/REVIEW.md B1). -->

- **What changed**:
- **How tested**:
- **Risks**:

## Track & Scope

- [ ] Evaluation & Benchmarking (`tests/evals/`)
- [ ] Ariadne Web UX (`web/src/`)
- [ ] Workflows, Providers & CLI DX (`graph/workflows/`, `providers/dialects/`, `cli/`, `docs/`)
- [ ] Platform Core (`core/`, `harness/`, `stages/`, `session/`, `src/olympus/web/`) — *contract aligned in linked Issue*

## Screenshots (required for Web UI changes)

<!-- Required for any PR touching web/ UI or visual behavior (docs/REVIEW.md B13). Attach real browser screenshots directly here (light and dark themes recommended). Write "N/A (no Web UI changes)" otherwise. -->

## Verification

- [ ] `uv run ruff check .`
- [ ] `uv run mypy src`
- [ ] `uv run pytest -q`
- [ ] Frontend checks (if touching `web/` or backend API schemas): `pnpm --dir web generate:api && pnpm --dir web typecheck && pnpm --dir web build`
- [ ] Real browser screenshots attached above (required for any `web/` UI or visual changes)
- [ ] Checked against [docs/REVIEW.md](docs/REVIEW.md) (`Closes #<issue-id>`, `write_scope`, and blocking criteria)
