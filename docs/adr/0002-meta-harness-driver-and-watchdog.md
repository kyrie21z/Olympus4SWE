# ADR-0002: Meta-Harness AgentDriver Seam, Post-Flight Diff Gate, and Two-Tier Watchdog

- **Status**: Accepted
- **Date**: 2026-11-21
- **Tracking**: #61 (`Diff Gate`), #62 (`AgentDriver` seam & config), #63 (Live milestone observability), #64 (`PiDriver` & heterogeneous `Argus`), #65 (Two-tier `Watchdog`)
- **Phasing**: Phase 1 (`AgentDriver` seam + live milestone observability + `Diff Gate`) → Phase 2 (`PiDriver` + `Watchdog`, with `ACPDriver` seam reserved)

## Context

Olympus4SWE's core differentiator is its deterministic SDLC control plane (spec alignment, hash-bound approvals, disjoint-scope DAG scheduling, cold-start verification, isolated delivery checks, and event-sourced HITL controls)—not re-inventing a general-purpose coding loop. Mature standalone coding harnesses (Pi Agent, Claude Code, Codex) increasingly commoditize tool loops, parallel tool execution, compaction, and vendor-tuned coding ergonomics.

At the same time:
1. Default global token hard caps were removed (PR #59) because real projects vary widely in token consumption. Without hard caps, long-running tasks need **live in-flight observability** and **semantic progress monitoring** to detect stuck loops, hangs, or scope drift instead of running blind.
2. Stage 1 structured nodes (`Align`, `Design`, `Plan`), fast offline unit tests, CI, and portability to the sister repository `SWE-Olympus` still require a zero-external-CLI, pure-Python execution path.
3. Copying only `write_scope`-matching files back from a sandbox without checking for out-of-scope edits risks silent integration failures when a worker (or `bash`) modifies shared contracts or files outside its declared scope.

## Decision

Position Olympus4SWE as a **Harness of Harnesses** by decoupling the deterministic control plane from execution loops behind four architectural invariants:

1. **Unified `AgentDriver` Contract (Keep Native, Plug External)**:
   - Define a narrow driver contract supporting `run_turn`, live milestone callbacks, mid-turn `steer`, cooperative/forced `cancel`, and standardized `Turn Receipt` + trace output.
   - Retain the existing Python loop as `NativeDriver` (also implementing `steer()` by injecting guidance on the next step) for Stage 1 structured nodes, offline tests, and default fallback.
   - Land `PiDriver` first using `pi --mode rpc` persistent pipes (supporting live events, `steer`, and clean/forced termination), falling back to `--mode json` for older builds, while keeping the contract shaped cleanly for a subsequent `ACPDriver` (Claude Code, Codex, and other ACP agents).

2. **Role-Scoped Heterogeneous Driver Configuration**:
   - External drivers apply to **Stage 2 Coding Workers (`Hephaestus`)** and optionally **Stage 2 Independent Verifiers (`Argus`)**, enabling heterogeneous cross-vendor review (e.g., Pi/Codex implements, another model/driver reviews). Stage 1 (`Align`/`Design`/`Plan`) stays on lightweight `NativeDriver`.
   - `worker_driver`, `reviewer_driver`, and `monitor_model` are independently configurable via CLI flags, environment variables, and the Ariadne run-creation form, defaulting to `NativeDriver` on the primary provider.

3. **OS/Git-Boundary Determinism via Post-Flight `Diff Gate`**:
   - Pre-flight tool hooks remain optional per driver (native policy guard or driver extension), never the sole enforcement layer.
   - Before acceptance checks or overlay promotion, a deterministic **Diff Gate** inspects every tracked and untracked change in the sandbox against `write_scope` and `forbidden_globs`. Any violation rejects and requeues the task with the exact offending paths rather than silently dropping files.

4. **Two-Tier Live Observability and Two-Tier `Watchdog`**:
   - **Observability split**: Low-frequency, privacy-safe milestone events (`tool_started`, `tool_finished`, `watchdog.checked`, `watchdog.steered`, `turn.receipt`) persist to SQLite `events` so Ariadne's existing SSE stream renders live progress and survives page reloads; high-frequency `text_delta` streams and raw tool outputs stay in `worker.jsonl` plus an in-memory latest-line preview.
   - **L1 deterministic triggers (0-token)**: Watch the event stream for 3 consecutive identical tool/command errors, 180s silence, or a 300s periodic soft check. Healthy short turns trigger zero monitor LLM calls.
   - **L2 small-model `Watchdog` (read-only)**: When L1 triggers, evaluate a compact snapshot (task objective, `write_scope`, recent event/error tail, `git diff --stat`) and return one of `HEALTHY`, `STUCK_LOOP`, `SCOPE_DRIFT`, or `BLOCKED_EXTERNAL` plus a one-line human summary:
     - `HEALTHY`: emit summary to Ariadne.
     - First `STUCK_LOOP` or `SCOPE_DRIFT`: deliver gentle `steer()` guidance (capped at 2 nudges per attempt).
     - Consecutive anomalies (2×) or `BLOCKED_EXTERNAL`: transition through `RunControl` to paused/`blocked` for human decision, preserving sandbox artifacts.
     - Watchdog errors or timeouts degrade silently (`watchdog.error`) and never abort an active worker.

## Alternatives Considered

1. **Replace `NativeLoop` entirely with Pi Agent or ACP**:
   - *Rejected*: Breaks pure-Python zero-Node CI and offline `MockLLMProvider` unit tests, adds subprocess overhead to short structured-JSON stages (`Align`/`Design`/`Plan`), and impairs portability to `SWE-Olympus`.
2. **Enforce `write_scope` only through per-harness pre-tool hooks**:
   - *Rejected*: Forces maintaining fragile hook dialects across Pi, Claude Code, and Codex (including trust/config quirks) and still cannot statically guarantee arbitrary `bash` commands stay in scope.
3. **Auto-revert (`git checkout`) out-of-scope edits and accept remaining diff**:
   - *Rejected*: Risks accepting code whose sandbox tests only passed because of the reverted out-of-scope edit; explicit rejection forces the worker to fix the design or surface a contract gap.
4. **Unconditional periodic small-model polling with direct kill authority**:
   - *Rejected*: Wastes calls during healthy progress, risks false kills during long builds or deep reasoning, and lets a weaker model unilaterally destroy a stronger model's work.
5. **Persist every streaming `text_delta` chunk to SQLite `events`**:
   - *Rejected*: Causes severe SQLite WAL write amplification and bloats the audit event stream with transient token fragments.

## Consequences

- **Positive**: Turns Olympus4SWE into a true Meta-Harness capable of heterogeneous multi-vendor coding and review; closes the silent out-of-scope bug for all drivers; gives operators live progress visibility and semantic stall protection without brittle static token caps.
- **Negative**: Introduces a thin driver lifecycle layer (process management, RPC framing, and thread-safe `steer`/`cancel` coordination) and optional external CLI dependencies (`pi`, later ACP servers) when external drivers are enabled.
- **Mitigation**: Keep `NativeDriver` as the default zero-dependency baseline; test `PiDriver` and `Watchdog` deterministically via fake-CLI/mock-provider seams; roll out in two vertical slices (Phase 1 core seam + Diff Gate + live milestones; Phase 2 `PiDriver` + `Watchdog`).

## Amendment (2026-11-22)

Following the BlackNote Dev longitudinal E2E run (tracking #83–#87), two Watchdog & sandbox operational defaults were refined without relaxing ADR-0002's safety invariants:

1. **Anomaly-driven Watchdog default (`interval_seconds = None`)**: Blind periodic L2 polling is disabled by default (remaining opt-in when callers set a positive `interval_seconds`). L1 now triggers L2 strictly on deterministic 0-token anomalies: 3× repeated tool errors, `180s` silence, repeated context compactions (`compaction_threshold = 3`), or newly observed out-of-scope sandbox mutations relative to the task's initial baseline. Every L2 evaluation is settled into `RunStore` `BudgetLedger` (`actor = "watchdog"`).
2. **Stateful sandbox retention on retry**: Retried attempts reuse the task's sandbox anchored to its persisted Attempt-1 `SandboxBaseline` (including original `base_commit`) for targeted in-place repair rather than wiping valid `write_scope` work. `Diff Gate` still never auto-reverts out-of-scope files.
