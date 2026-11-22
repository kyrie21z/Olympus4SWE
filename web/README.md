# Ariadne web console

Ariadne is a React 19 and TypeScript SPA served by FastAPI. The Python runtime
serves the committed Vite output from `src/olympus/web/static/` and requires no
Node.js installation.

## Develop and build

```bash
pnpm --dir web install --frozen-lockfile
pnpm --dir web generate:api
pnpm --dir web typecheck
pnpm --dir web build
```

For hot reload, start `uv run olympus serve` and `pnpm --dir web dev` in separate
terminals. Open the URL printed by Vite. Vite proxies `/api` and `/healthz` to
FastAPI at `127.0.0.1:8765`.

Regenerate `src/api/schema.d.ts` when `src/olympus/web/openapi.json` changes.
Rebuild and commit the generated static assets after frontend source changes.
CI checks both artifacts for drift. The
[design-system tests](../tests/unit/test_design_system.py) also verify the
bundle source hash and the Warm Editorial Stone token rules.

The topology view loads React Flow and ELK on demand in the
`ariadne-graph` chunk. The workflow and task dependency graphs share
[GraphCanvas](src/graph/canvas/GraphCanvas.tsx) and
[layout helpers](src/graph/layout/shared.ts).
