# ADR-0004: Native Multi-Workflow Architecture and Distilled `t3-pstack` Primitives

- **Status**: Accepted
- **Date**: 2026-11-28
- **Tracking**: #188 (Platform Core workflow decoupling), #189 (`pstack_craft` & 6 orthogonal primitives), #190 (`pstack_bugfix` & read-only `investigate`), #191 (`pstack_arena` & `pstack_hillclimb`)
- **Related**: #187 (OpenCode external `AgentDriver` adapter), [ADR-0002](0002-meta-harness-driver-and-watchdog.md)
- **Phasing**: Phase 0 (#188 Core decoupling) → Phase 1 (#189 `pstack_craft` + live eval) → Phase 1.5 (#190 `pstack_bugfix` & `investigate`) → Phase 2 (#191 `pstack_arena` & `pstack_hillclimb`, unlocked with #187 multi-model drivers)

## Context

Olympus4SWE's declarative workflow layer (`src/olympus/graph/`) was originally built around a single contract-waterfall paradigm (`GoalSpec → DesignSpec → TaskPlan → execute_task_dag`). A codebase audit of `main` surfaced five structural rigidities that prevent landing fundamentally different engineering workflows alongside the existing SDLC flows:

1. **Single-paradigm workflow variants**: All three built-in workflows (`full_sdlc_tdd`, `product_blueprint`, `repo_fast_patch`) share the same waterfall backbone; even `repo_fast_patch` still runs `align_goal → pm_decompose → validate_plan → execute_task_dag` with `auto_approve: true`.
2. **Hardcoded upstream artifact coupling**: `OrchestrateStage.run_execution` (`src/olympus/stages/orchestrate.py:3532-3554`) requires an approved `TASK_PLAN` bound to an upstream `GoalSpec`, and `DesignStage.propose_design` requires a confirmed `GoalSpec`. Non-waterfall workflows cannot execute without fabricating dummy specs and task DAGs.
3. **Missing verification/review/iteration primitives and implicit Pantheon gods**: `src/olympus/graph/loader.py:38-48` exposes only 9 node types. `Momus` (adversarial review) and `Argus` (real-surface verification) have no first-class graph nodes (`verify_product_delivery` is buried inside `execute_task_dag`; `critic_parallel` is a no-op under default `enable_critique=False`). `_NODE_GODS` (`src/olympus/graph/progress.py:761-774`) and `web/src/lib/workflowStages.ts:57-77` map gods and UI stages by hardcoded literal node names.
4. **Hash-inferred resume cursor**: `resolve_resume_node` (`src/olympus/graph/progress.py:90-164`) reconstructs pause/resume position from the three SDLC approval hashes rather than reading a persisted cursor, making non-SDLC pause/resume impossible.
5. **Static 3-slot driver config & Web/CLI context asymmetry**: `ExecutionDriversConfig` (`src/olympus/harness/driver.py:98`) only defines `stage_driver`, `worker_driver`, and `reviewer_driver`, while headless `src/olympus/session/runner.py:673-690` drops `workspace_mode`, `task_type`, and `scope_paths` that Web `supervisor.py` injects.

Meanwhile, practical brownfield engineering with `t3-pstack` (`poteto-mode`) demonstrates that autonomous-by-default, evidence-first loops (reproduce before fixing, two-shape architecture sketches, diff hygiene, cold-start run-proven adversarial interrogation, competing worktree arenas with grafting, and frozen-harness metric hill-climbing) consistently outperform heavy upfront spec gates when intent is already clear. However, in an external prompt plugin those disciplines remain probabilistic and drift over long runs.

Olympus4SWE's deterministic control plane can elevate those prompt disciplines into **mechanically enforced invariants**—while keeping existing SDLC workflows intact as independent peers ready for a future Master Controller Agent to select across an expanding workflow catalog.

## Decision

Adopt a **Three-Layer "Skill Workflow → Native Workflow" Architecture** that keeps all workflows strictly independent, backward-compatible, and composable from a compact set of deterministic primitives.

### 1. Independent Coexistence & Self-Describing Workflow Manifests (No Auto Router Yet)

- Existing workflows (`full_sdlc_tdd`, `product_blueprint`, `repo_fast_patch`) stay untouched in execution semantics and continue to run independently.
- Do **not** build an automatic workflow classifier (`Auto` router) in Phase 1. Workflows are selected explicitly today; a future Master Controller Agent will select across SDLC, `pstack`, and additional future workflows by reading each workflow's declarative manifest.
- Extend `WorkflowDefinition` and `WorkflowNodeDef` (`src/olympus/graph/loader.py`) with purely additive, backward-compatible fields:
  - Workflow level: `family` (`"sdlc" | "pstack" | "minimal" | "forensics" | ...`), `autonomy` (`"gated" | "autonomous_hard_stops" | "read_only"`), `applicable_scenarios: list[str]`, `anti_patterns: list[str]`, and `stop_after` extended with `"verify"` and `"report"`.
  - Node level: `god` (`prometheus | athena | hephaestus | momus | argus`), `phase`, `role`, and `skill_refs: list[str]`.
- `src/olympus/graph/progress.py` and `web/src/lib/workflowStages.ts` resolve Pantheon god and UI phase from node metadata first, falling back to legacy literal-name maps only for historical workflows.

### 2. Explicit Resume Cursors & Cohesive Control-Plane Primitives

- Persist `(paused_node, pause_reason)` explicitly in SQLite when `GraphPaused` is raised. `resolve_resume_node` checks the persisted cursor first and falls back to legacy 3-hash inference only for historical runs.
- Keep `src/olympus/graph/engine.py` minimal and readable: bounded retry/iteration loops use existing `max_visits` + conditional `route` back-edges; multi-worktree competition (`arena_race`) and frozen-harness iteration (`hillclimb_loop`) are encapsulated as self-contained deterministic node handlers backed by `WorktreeManager` and `DriverRegistry`, avoiding scatter/gather AST complexity in `engine.py`.
- Propagate `workspace_mode`, `task_type`, and `scope_paths` consistently in headless `src/olympus/session/runner.py`.

### 3. Distilling `t3-pstack`: Control-Plane Primitives vs. Skill Packs vs. Supervisor Services

Rather than porting `t3-pstack`'s 22 playbooks and 67 skills 1:1 (which would cause module fragmentation—see Doc2Harness empirical finding `r = -0.423` between module count and pass rate), separate concerns into three buckets:

1. **Deterministic Native Primitives (6 core + 2 Phase 2)** backed by four lightweight `FingerprintedModel` contracts in `src/olympus/core/contracts.py` (`EvidenceBrief`, `ArchitectSketch`, `InterrogateReport`, `SurfaceVerificationReport`) and implemented in a single cohesive stage module (`src/olympus/stages/craft.py`):
   - `explore_reproduce` (`prometheus`, `read_only=True`, `write_scope=[]`): repo/runtime evidence gathering; when `require_repro=true`, deterministically verifies `repro_command` exits non-zero on `base_ref` before advancing.
   - `architect_sketch` (`athena`, `read_only=True`): enforces `>=2` candidate structural shapes, single state ownership, verification plan, and `write_scope` lock for downstream `DiffGate`.
   - `build_direct` (`hephaestus`): isolated `git worktree` + `DiffGate` + `clean_ephemeral_artifacts` directly consuming `ArchitectSketch` / `EvidenceBrief` (no `TaskPlan` DAG required); supports `failing_test_first=true` and incremental repair from `InterrogateReport`.
   - `hygiene_pass` (`hephaestus`): diff-scoped `deslop` / comment hygiene cleanup followed by behavioral re-check.
   - `interrogate_review` (`momus`, `role: interrogator`): cold-start fresh session auditing `base..HEAD` diff; deterministic filter enforces run-proven sandbox receipts for blocking findings; optional dual-lane `thermo` (Correctness/Security + Code-Judo); routes back to `build_direct` up to `max_visits=3`.
   - `verify_surface` (`argus`, `role: verifier`): first-class standalone real-surface verification node emitting `SurfaceVerificationReport` where `inconclusive` never counts as pass.
   - *(Phase 2)* `arena_race` (`athena` + `hephaestus`): N parallel isolated worktrees → rubric-scored independent Judge → Graft → cleanup.
   - *(Phase 2)* `hillclimb_loop` (`hephaestus`): anti-tamper frozen benchmark harness → 1 grounded hypothesis/iteration → deterministic keep or `git reset --hard` → SQLite `decision_trail`.
2. **Progressive Skill Packs (`src/olympus/skills/pstack/*.md`)**: Distilled reasoning principles (`attack-the-premise`, `subtract-before-add`, `code-judo`, `deslop`, `no-comments`) loaded via `node.skill_refs` into stable system-prompt prefixes (preserving KV-cache hits).
3. **Excluded from Workflow YAML Layer (Supervisor / Delivery Scope)**: Host and GitOps operational playbooks (`babysit`, `shipping`, `autopilot-full`, `autopilot-stack`, `worktree-cleanup`, `pause-safely`, `session-pickup`, `opening-a-pr`) belong to `Supervisor` / delivery lifecycle services, not problem-solving workflow topologies.

### 4. Single-Model Default with Cold-Start Isolation + `role_drivers` Slot

- Extend `ExecutionDriversConfig` (`src/olympus/harness/driver.py`) with an optional `role_drivers: dict[str, NonEmptySetting]` mapping and `resolve_driver_for_role(role)`.
- **Phase 1 default (single model)**: Missing role entries fall back cleanly to `stage_driver` / `worker_driver` / `reviewer_driver`, while the control plane strictly enforces **zero-history cold-start sessions** for `interrogator` and `verifier` (preventing builder confirmation bias without blocking single-model development or offline evals).
- **Multi-model unlock (#187 OpenCode)**: Once OpenCode and additional external drivers land, heterogeneous cross-family panels (`interrogator`, `arena_runners`, `arena_judge`) activate purely via configuration with zero graph-engine rewrites.

### 5. Three-Axis Orthogonal Workflow Taxonomy (6 + 1 Portfolio)

Workflows are partitioned along three epistemic axes—**Intent Certainty × Solution/Verification Structure × Autonomy Budget**—so each workflow has a clear, non-overlapping mandate:

| Workflow ID | Family | Autonomy | Applicable Engineering / Product Scenarios |
| --- | --- | --- | --- |
| `full_sdlc_tdd` *(existing)* | `sdlc` | `gated` (3 gates) | Greenfield 0→1, ambiguous product ideas, large multi-module systems requiring upfront black-box `GoalSpec`, white-box `DesignSpec`, and disjoint parallel Task DAGs |
| `product_blueprint` *(existing)* | `sdlc` | `gated` (`stop_after: plan`) | Pure RFCs, architecture blueprints, multi-phase roadmaps (`spec_only` / `blueprint`, zero code writes) |
| `repo_fast_patch` *(existing, narrowed)* | `minimal` | `autonomous_hard_stops` | Known-cause trivial patches (1–2 files, known fix location, lowest latency/token overhead) |
| `pstack_craft` *(#189)* | `pstack` | `autonomous_hard_stops` | Brownfield multi-file features, signature changes, cross-boundary refactors with clear intent |
| `pstack_bugfix` *(#190)* | `pstack` | `autonomous_hard_stops` | Unknown-cause bugs, regressions, non-obvious defects requiring reproduction and failing-test-first proof |
| `investigate` *(#190)* | `forensics` | `read_only` | Read-only root-cause forensics, runtime/trace analysis, "why does this happen?" without workspace edits |
| `pstack_arena` *(#191)* | `pstack` | `autonomous_hard_stops` | Contested high-leverage architectures or refactors benefiting from parallel worktree competition + Judge + Graft |
| `pstack_hillclimb` *(#191)* | `pstack` | `autonomous_hard_stops` | Measurable performance optimization, latency/build reduction, visual parity, or benchmark hill-climbing |

## Alternatives Considered

1. **Single monolithic adaptive workflow (`poteto_adaptive.yaml`) with internal playbook routing**:
   - *Rejected*: Forces 25+ nodes into one YAML, degrades Ariadne topology readability, conflates workflow selection (future Master Controller Agent responsibility) with execution topology, and prevents clean per-workflow ablation in `tests/evals/`.
2. **Prompt-only skill injection into the existing 3 SDLC YAMLs**:
   - *Rejected*: Cannot bypass the hardcoded `GoalSpec → DesignSpec → TaskPlan` approval chain or mechanically enforce run-proven interrogation receipts, failing-repro-before-fix gates, or deterministic keep/revert hill-climbing.
3. **Direct 1:1 port of all 22 `t3-pstack` playbooks and 67 skills as graph nodes**:
   - *Rejected*: Blurs SWE problem-solving topologies with GitOps/PR lifecycle management and introduces severe module fragmentation.

## Consequences

- **Positive**:
  - Zero behavioral regression to existing SDLC workflows (`full_sdlc_tdd`, `product_blueprint`, `repo_fast_patch`).
  - Elevates `t3-pstack`'s highest-leverage engineering habits into deterministic, SQLite-audited control-plane invariants while making `Momus` and `Argus` first-class graph nodes.
  - Establishes a reusable, manifest-driven pattern for onboarding *any* future external or custom skill workflow and prepares cleanly for both the OpenCode multi-model driver (#187) and a future Master Controller Agent.
- **Negative**:
  - Introduces additive Platform Core changes across `graph/`, `core/contracts.py`, `harness/driver.py`, and `web/src/lib/workflowStages.ts` (#188).
  - Adds a second artifact track (`EvidenceBrief` / `ArchitectSketch`) alongside `GoalSpec` / `DesignSpec` / `TaskPlan`.
- **Mitigation**:
  - Strictly phase delivery (#188 → #189 → #190 → #191) behind Eval-First gates: #189 must run head-to-head benchmarks on `SWE-Interact` / `Harbor-Interact` (`pstack_craft` vs `full_sdlc_tdd` vs `repo_fast_patch`) measuring Pass Rate, Token Cost, Wall Time, and Human Interrupts, and #191 must demonstrate marginal gain justifying multi-worktree cost before default adoption.
  - Keep all six Phase 1 primitives inside one cohesive module (`src/olympus/stages/craft.py`) and retain `core/` purity enforced by `tests/unit/test_architecture_invariants.py`.
