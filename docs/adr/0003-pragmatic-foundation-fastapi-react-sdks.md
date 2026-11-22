# ADR-0003: Pragmatic Build-vs-Buy Foundation — FastAPI + OpenAPI, Unified React/TS Ariadne SPA, and Official LLM SDK Adapters

- **Status**: Accepted
- **Date**: 2026-11-21
- **Supersedes**: [ADR-0001](0001-ariadne-graph-island-reactflow-elk.md)
- **Tracking**: #73 (FastAPI + OpenAPI contracts), #74 (Unified React/TS Ariadne SPA + CI guardrails), #75 (Official `openai`/`anthropic` SDK thin dialect adapters)
- **Spec**: `.specs/pragmatic-foundation-refactor/spec.md`
- **Phasing**: PR 1 (FastAPI + strict Pydantic OpenAPI contract) → PR 2 (Unified React + TypeScript + Vite Ariadne SPA + CI guardrails) → PR 3 (Official `openai` / `anthropic` SDK thin dialect adapters)

## Context

Early `Olympus4SWE` milestones minimized runtime dependencies (5 packages in `pyproject.toml`) to preserve transparency and avoid heavy orchestration frameworks. Over subsequent milestones, however, treating "minimal dependencies" as synonymous with "first principles" created significant self-inflicted infrastructure tax:

1. **Hand-rolled stdlib HTTP/SSE server**: `src/olympus/web/server.py` (725 LOC) manually implements routing, query parsing, static MIME serving, error handling, and threaded SSE streaming on top of `http.server.ThreadingHTTPServer`, while `projection.py` (1,610 LOC) emits untyped `dict[str, Any]` payloads with no machine-checked API schema.
2. **Dual-track Ariadne frontend (ADR-0001's compounding cost)**: After ADR-0001 introduced `React 19 + TypeScript + Vite + pnpm` for `web-graph/`, the console ended up paying the toolchain cost of a modern frontend build while keeping ~2,850 LOC of untyped vanilla JavaScript (`app.js`, `ui.js`) and ~2,260 LOC of monolithic CSS (`style.css`) for the main shell.
3. **Hand-rolled LLM HTTP wire dialects**: `src/olympus/providers/dialects/` manually serializes and parses OpenAI Chat Completions and Anthropic Messages over raw `httpx.post`.
4. **"The Codebase is Memory" in AI-assisted development**: When coding agents iterate on a repository, existing files act as their strongest behavioral template. Leaving a 2,657-line untyped JS monolith and untyped dict HTTP endpoints in the repository compounds low-quality patterns on every future feature, whereas statically checked boundaries (`mypy --strict`, Pydantic → OpenAPI → TypeScript, CI lint gates) establish a deterministic **Pit of Success**.

First-principles engineering optimizes core domain constraints and iteration leverage—not re-inventing commoditized infrastructure from scratch.

## Decision

Establish an explicit three-tier **Build-vs-Buy** boundary and refactor the three infrastructure layers in three sequential, green-main PRs without altering the stable core domain modules (`core/`, `stages/`, `graph/`, `harness/`):

### 0. Three-Tier Build-vs-Buy Boundary
- **Build & own in-house (Core Moat)**: Deterministic SDLC control plane (`core/`), two-stage contracts & triage (`stages/`), blackboard workflow engine (`graph/`), and meta-harness driver / diff gate / watchdog (`harness/`).
- **Adopt mature libraries (Commoditized Infrastructure)**: `FastAPI` + `uvicorn` for Web HTTP/SSE and OpenAPI generation; unified `React + TypeScript + Vite` for the Ariadne console; official `openai` and `anthropic` Python SDKs behind thin dialect adapters.
- **Continue rejecting (Black-Box Orchestration Frameworks)**: LangGraph, CrewAI, AutoGen, and opaque LLM meta-routers (e.g., LiteLLM) that take over control flow, prompts, state transitions, or credential failover policy.

### 1. FastAPI Backend with Full Pydantic Contracts & OpenAPI Export (PR 1)
- Completely remove `ThreadingHTTPServer` and `AriadneRequestHandler`; expose `create_fastapi_app(app: AriadneApp) -> FastAPI` and serve `olympus serve` via `uvicorn`.
- Define strict Pydantic v2 models for **every** API request (including a discriminated union for HITL commands) and **every** response (including full `RunProjection`, events, providers, workflows, demos, and task artifacts).
- Register custom FastAPI exception handlers preserving existing wire status codes and structured envelopes (`400 ErrorResponse` for validation/`ValueError`, `404 ErrorResponse` for `NotFoundError`, `409 ConflictResponse` with fresh `projection` for `ConcurrencyConflictError`), documented directly in OpenAPI.
- Keep the existing SSE wire contract (`event: state`, `event: activity`, `?once=1`, `: heartbeat`) backed by 0.5s SQLite polling so external CLI mutations remain immediately visible without introducing an IPC event bus.
- Commit the generated `openapi.json` snapshot and enforce zero drift in Python tests.

### 2. Unified React + TypeScript + Vite Ariadne SPA (PR 2, Superseding ADR-0001)
- Supersede ADR-0001's Graph Island split by promoting `web-graph/` to `web/`, migrating the entire Ariadne console (left thread rail, stage pipeline, HITL banners & approval reviews, clarify panel, all 4 views, resizable inspector dock, and both native `<dialog>` modals) into typed TSX in a single clean cutover, and deleting `app.js`, `ui.js`, and legacy static shell files.
- Generate frontend TypeScript API types directly from `openapi.json` using `openapi-typescript` and consume them through thin typed `fetch` + SSE hooks (no Redux, Zustand, or TanStack Query).
- Preserve `tokens.css` and the Warm Editorial Stone design system verbatim using modular CSS and native `<dialog>` modals; migrate `test_design_system.py` to statically lint `web/src/**`.
- Load the ELK + React Flow graph module via a code-split `React.lazy` chunk wrapped in a React `ErrorBoundary`, keeping `elk.bundled.js` out of the initial bundle and preserving fault isolation when graph rendering or chunk loading fails.
- Continue committing the pre-built static output into `src/olympus/web/static/` with a `source-sha256` freshness stamp verified by Python tests so `uv run olympus serve`, wheel installs, and pure-Python `pytest` remain **zero-Node at runtime**.
- Add a GitHub Actions `frontend` CI job enforcing `tsc --noEmit`, OpenAPI↔TS schema freshness, `vite build` / `source-sha256` cleanliness, and a hard ban on `.js`/`.jsx` source files under `web/src/` and legacy static JS files.

### 3. Official LLM SDK Thin Dialect Adapters (PR 3)
- Keep `LLMProvider`, `PooledProvider`, `CredentialPool`, `ProviderRegistry`, and `OpenAIChatCompat` untouched.
- Replace raw `httpx.post` transport inside `OpenAIChatDialect` and `AnthropicMessagesDialect` with thin adapters around `openai.OpenAI(..., max_retries=0)` and `anthropic.Anthropic(..., max_retries=0)`.
- Disable SDK automatic retries so `CredentialPool` remains the sole deterministic owner of multi-key/multi-endpoint sticky failover; map SDK `APIStatusError` cleanly to `HttpStatusError`.
- Maintain strict 1:1 behavioral parity (no streaming, prompt-caching, or reasoning interface expansion in this refactor) and inject `httpx.MockTransport` via SDK `http_client` for deterministic offline tests.

## Alternatives Considered

1. **Keep ADR-0001's Graph Island + vanilla JS console + stdlib `ThreadingHTTPServer`**:
   - *Rejected*: Retains an untyped 2,657-line JS monolith and untyped dict HTTP layer as permanent slop attractors despite already paying the React/Vite/pnpm toolchain cost.
2. **Incremental strangler-fig frontend migration across multiple PRs**:
   - *Rejected*: Requires maintaining temporary imperative JS↔React bridges across every panel when the console is bounded (~2.6k LOC) and already guarded by a 1,606-line Playwright browser suite.
3. **Shallow FastAPI wrapper keeping `dict[str, Any]` responses**:
   - *Rejected*: Emits `Record<string, unknown>` in OpenAPI, forfeiting the primary compile-time guardrail against backend/frontend field drift.
4. **Adopt LiteLLM for providers or Tailwind/shadcn + Zustand/TanStack Query for frontend**:
   - *Rejected*: LiteLLM brings heavy transitive dependencies and opaque routing/retry behavior that competes with `CredentialPool`; TanStack Query, Zustand, and a third-party dialog library add unnecessary abstraction over a single-server SSE workbench with two native `<dialog>` modals.
5. **Exclude pre-built frontend assets from git (require Node at runtime)**:
   - *Rejected*: Breaks zero-Node `uv run olympus serve`, wheel distribution, and pure-Python test execution.

## Consequences

- **Positive**: Closes the last major untyped blind spot in the repository; establishes end-to-end compile-time type safety from Python Pydantic models through OpenAPI to React TSX; eliminates stdlib HTTP/SSE plumbing and dual-track frontend maintenance; delegates LLM wire protocol upkeep to upstream official SDKs while preserving full control over core SDLC orchestration.
- **Negative**: Adds runtime Python dependencies (`fastapi`, `uvicorn`, `openai`, `anthropic`) and a Node/`pnpm` CI job alongside committed SPA static build artifacts.
- **Mitigation**: Keep core modules (`core/`, `stages/`, `graph/`, `harness/`) strictly free of web/SDK imports so `SWE-Olympus` portability is unaffected; verify bundle freshness and OpenAPI↔TS synchronization automatically in CI; deliver in 3 independent green-main PRs.
