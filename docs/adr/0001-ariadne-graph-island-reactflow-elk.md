# ADR-0001: Adopt React Flow + ELK Pre-Built Graph Island for Ariadne Flowcharts

- **Status**: Superseded by [ADR-0003](0003-pragmatic-foundation-fastapi-react-sdks.md)
- **Date**: 2026-11-21
- **Tracking**: #56 (Task Dependency Graph cut-over), #57 (Orchestration Workflow cut-over)

## Context

Ariadne is served by a lightweight Python stdlib HTTP/SSE server (`src/olympus/web/server.py`) hosting vanilla HTML/CSS/JS static assets with zero Node runtime requirement. Its Topology view renders two directed graphs: the **Orchestration Workflow** (up to 11+ nodes with conditional skip/retry routes) and the **Task Dependency Graph** (multi-layer DAGs with skip-layer dependencies, critical paths, and blocked propagation).

Hand-rolled SVG string concatenation led to compounding defects: opposite scale distortion across graphs, `strokeWidth`-scaled oversized arrowheads, skip-layer edges passing behind intermediate nodes, stacked entry ports, and no pan/zoom or lineage focus ergonomics.

## Decision

Introduce a self-contained **Graph Island** built with **`@xyflow/react` (React Flow)** + **`elkjs` (`elk.bundled.js`, ELK Layered algorithm)**, authored in a dedicated `web-graph/` directory (`pnpm` + Vite library build, IIFE format) and committed as a pre-built static bundle under `src/olympus/web/static/graph/` (`ariadne-graph.js`, `ariadne-graph.css`).

- The existing Ariadne console shell remains vanilla JS/CSS and communicates with the island through a narrow imperative mount API (`window.AriadneGraph.mountDag` / `mountWorkflow`), each returning a `{ update(props), fitView(), destroy() }` handle. The host passes plain JSON view models; the island never imports backend internals.
- Nodes render as real HTML/CSS cards inheriting `tokens.css` custom properties (light/dark parity, crisp CJK + monospace typography, keyboard/DOM accessibility). The island stylesheet holds no colour literals.
- `elk.layered` computes crossing-minimized coordinates, obstacle-avoiding orthogonal edge polylines, and distributed side ports completely offline in-browser without external worker/WASM files. Task DAG columns are pinned to the projection's `node.layer` waves.
- If ELK fails, or the input contains a cycle, the island falls back to a deterministic cycle-safe layered layout instead of rendering blank. If the bundle itself fails to load, the host shows a notice and the rest of Ariadne keeps working.
- Pre-built assets are committed so `uv run`, `pytest`, and wheel installs remain zero-Node at runtime. The bundle carries a `source-sha256` stamp over `web-graph/` sources; a Python unit test recomputes it so a stale bundle fails CI without needing Node.

## Alternatives Considered

1. **Zero-dependency custom SVG + custom Sugiyama/routing engine in vanilla JS**:
   - *Rejected*: Keeps zero frontend build chain, but re-implementing robust orthogonal obstacle avoidance, distributed port routing, pan/zoom gestures, and smooth focus transitions from scratch carries high maintenance cost and edge-case fragility compared to ELK + React Flow.
2. **Cytoscape.js (+ dagre/elk) on HTML5 Canvas**:
   - *Rejected*: Avoids React, but `<canvas>` rendering does not natively consume `tokens.css` variables, weakens rich multi-row CJK card layout, and drops native DOM keyboard focus and Playwright DOM locator ergonomics.
3. **Whole-console migration to React + Vite**:
   - *Rejected*: Unnecessary blast radius (rewriting ~2,700 lines of working approval, SSE, matrix, waterfall, and dock code) when only the two graph canvases require a specialized graph engine.
4. **Dagre (`@dagrejs/dagre`) instead of ELK**:
   - *Rejected*: Smaller bundle, but weaker multi-layer obstacle avoidance and port separation — the primary visual failure mode in our multi-layer DAGs.

## Consequences

- **Positive**: High-craft editorial card rendering with automatic light/dark token inheritance; mathematically clean orthogonal obstacle-avoiding routing and port spacing; preserves offline zero-Node Python runtime and existing vanilla console code.
- **Negative**: Adds a `pnpm` + Vite build step when modifying graph UI code and commits a pre-built static JS/CSS bundle (dominated by `elk.bundled.js`) into the repository.
- **Mitigation**: Keep `web-graph/` narrowly scoped to pure graph visualization behind the JSON view-model contract; rebuild with `pnpm --dir web-graph build`; the source-stamp unit test keeps bundle freshness verified in development and CI.
