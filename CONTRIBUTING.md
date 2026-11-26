# Contributing to Olympus4SWE

Welcome! **Olympus4SWE** is engineered as a real, usable Agent-Native Software Engineering system — with strict typing, deterministic guardrails, scientific evaluation benchmarks, and a live human-in-the-loop console.

This guide explains how our team collaborates cleanly so every contributor owns visible, high-impact work without breaking other tracks.

---

## 1. 5-Minute Local Setup

### Prerequisites
- Python **3.13+** and [`uv`](https://docs.astral.sh/uv/)
- Node.js **22+** and `pnpm` (only needed if working on `web/`)
- Git

### Install & Verify
```bash
# Backend dependencies
uv sync --locked

# Run the backend quality gates
uv run ruff check .
uv run mypy src
uv run pytest -q
```

If you are working on the Ariadne web console (`web/`):
```bash
pnpm --dir web install --frozen-lockfile
pnpm --dir web typecheck
pnpm --dir web build
```

---

## 2. How the Project Is Organized (Three Orthogonal Loops)

Instead of everyone editing the same core files at once, Olympus4SWE is structured around **three clean, orthogonal loops** connected by typed contracts:

| Loop / Track | Primary Paths | Focus & Ownership |
| --- | --- | --- |
| **Loop 1 · Platform Core** | `src/olympus/core/`, `harness/`, `stages/`, `graph/engine.py`, `graph/loader.py`, `graph/progress.py`, `session/`, `src/olympus/web/` | Deterministic kernel, state machine, sandboxing, verification, and backend OpenAPI contracts. Maintained by the Tech Lead (`@kyrie21z`) to preserve conceptual integrity and keep all downstream tracks unblocked. |
| **Loop 2 · Evaluation & Benchmarking** | `tests/evals/` | Scientific benchmarks, datasets, scoring rubrics, ablation experiments, and diagnostic reports that objectively measure system quality and drive Core iteration. |
| **Loop 3A · Ariadne Web UX** | `web/src/`, `tests/unit/test_web_*_browser.py` | Oracle Board & Ariadne Console UI/UX, workflow/DAG visualizations, responsive polish, and browser end-to-end tests consuming the typed OpenAPI contract (`web/src/api/schema.d.ts`). |
| **Loop 3B · Workflows, Providers & DX** | `src/olympus/graph/workflows/*.yaml`, `src/olympus/providers/dialects/`, `src/olympus/providers/registry.py`, `src/olympus/cli/`, `README.md`, `docs/` | Declarative YAML workflow templates, model dialect adapters & presets, CLI ergonomics (`olympus doctor` / UX), documentation, and real-repository end-to-end demos. |

> See [AGENTS.md](AGENTS.md) for the full repository map and [GLOSSARY.md](GLOSSARY.md) for canonical domain terms (`Oracle`, `GoalSpec`, `DesignSpec`, `TaskPlan`, `Argus`, `Watchdog`, etc.).

---

## 3. The Core ↔ Benchmark Flywheel (Evaluator Independence)

Evaluation and the Core Engine collaborate tightly — and **deliberately remain decoupled**:

- **Why separation matters**: Just like top research labs separate evaluation harnesses from model/engine implementation, keeping `tests/evals/` strictly outside `src/olympus/` prevents benchmark leakage, accidental overfitting, and test-coupled hacks.
- **One-way dependency**: `tests/evals/` calls public runtime interfaces (`AlignStage`, `OrchestrateStage`, `RunStore`, `build_provider`, CLI) and keeps its own schemas in `tests/evals/schemas.py`. Production code under `src/olympus/` never imports `tests`.
- **How feedback drives Core improvement**:
  1. **Run & Diagnose**: After a Core snapshot lands on `main`, the Evaluation track runs benchmarks and opens an **Eval Diagnostic & Telemetry Feedback** issue containing the quantitative scorecard, top failure taxonomy, and minimal repro commands (`case_id`).
  2. **Request Telemetry Cleanly**: If a benchmark needs a new read-only metric or event field (for example, on `TurnReceipt` or `RunStore`), note it in the Diagnostic issue rather than patching `core/` directly. The Tech Lead will expose the typed contract cleanly.
  3. **Iterate & Re-benchmark**: Core fixes land on `main`, and re-running the benchmark quantifies progress (`v0 → v1 → v2`) with clean, reproducible curves.

---

## 4. Standard Workflow: Issue → Branch → PR

1. **Start from an Issue**
   - Pick an existing Starter/Track issue or open a new one using the templates in `.github/ISSUE_TEMPLATE/`.
   - Always state your **Goal**, **Non-goals**, **Expected `write_scope`** (which directories/files you plan to touch), and **Acceptance criteria** before writing code.
2. **Need a Platform Core API or Field?**
   - If your track needs a new backend endpoint, OpenAPI field, or core hook, open a short Issue describing the contract you need. The Tech Lead will prioritize landing the clean core interface so your track can build on top without merge conflicts.
3. **Branch from `main`**
   - Never push directly to `main`.
   - Create a focused branch: `feat/<short-name>`, `fix/<short-name>`, `eval/<short-name>`, or `docs/<short-name>`.
4. **Use AI Coding Assistants Effectively**
   - Using Cursor, Claude Code, Codex, or other coding agents is encouraged!
   - Point your assistant to [AGENTS.md](AGENTS.md) before starting a task so it respects module boundaries, `mypy --strict`, and OpenAPI sync rules automatically.
5. **Run Local Verification Before Opening a PR**
   ```bash
   # Backend checks (required for all Python changes)
   uv run ruff check .
   uv run mypy src
   uv run pytest -q
   ```
   ```bash
   # Frontend checks (required when touching web/ or backend API schemas)
   pnpm --dir web generate:api
   pnpm --dir web typecheck
   pnpm --dir web build
   ```
6. **Open a Focused PR**
   - Follow `.github/pull_request_template.md`: include `Closes #<issue-id>`, and fill in **What changed**, **How tested** (commands run and observed results), and **Risks**.
   - Keep PRs small and strictly scoped to the agreed `write_scope`. External contributors (including HW3 tracks) must not modify `src/olympus/**/*.py` or `src/olympus/web/openapi.json` without a contract Issue pre-approved by `@kyrie21z` (see `B4` in [docs/REVIEW.md](docs/REVIEW.md)).
   - **Web UI screenshots (required)**: Any PR touching Web UI or visual behavior under `web/` must attach real browser screenshots directly in the PR description (light and dark themes recommended; see `B13` in [docs/REVIEW.md](docs/REVIEW.md)).
   - Mark the PR **Ready for review** (Draft PRs skip automated review) and wait for required CI checks (`check`, `frontend`, and `browser`) plus review under [docs/REVIEW.md](docs/REVIEW.md) to pass.
   - PRs are merged via **Squash and merge** to keep `main` history clean and revertible.

---

## 5. Code Review Norms

- **Canonical PR review standard**: Every PR is evaluated against [docs/REVIEW.md](docs/REVIEW.md), which defines blocking criteria (`B1` through `B14`), maintainer-only approval paths, required CI checks, and how automated `Hydrogen` review works alongside human reviewers and auto-merge.
- **Objective gates first**: Formatting (`ruff`), strict typing (`mypy`), OpenAPI synchronization, architecture dependency boundaries (`tests/unit/test_architecture_invariants.py`), design-system checks (`tests/unit/test_design_system.py`), and Playwright browser tests (`browser`) are enforced automatically by CI so review stays focused on design and correctness.
- **Visual evidence for Web UI changes**: Any PR touching Web UI or visual behavior under `web/` must attach real browser screenshots (light and dark themes recommended) directly in the PR description before approval.
- **Clear severity**: Reviewers distinguish **`[Blocking]`** items (see `B1` through `B14` in [docs/REVIEW.md](docs/REVIEW.md)) from **`[Nit]`** suggestions (optional style polish).
- **Respectful and fast**: Highlight what works well, keep feedback actionable, and unblock teammates quickly.
