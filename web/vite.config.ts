import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const OUT_DIR = join(ROOT, "..", "src", "olympus", "web", "static");
const API_TARGET = "http://127.0.0.1:8765";

/**
 * Files whose bytes define the bundle. tests/unit/test_design_system.py mirrors this list, so a
 * source edit without `pnpm --dir web build` fails pytest offline, with no Node at runtime.
 */
const STAMP_FILES = [
  "package.json",
  "pnpm-lock.yaml",
  ".pnpmfile.cjs",
  "tsconfig.json",
  "vite.config.ts",
  "index.html",
  "../src/olympus/web/openapi.json",
];
const STAMP_DIRS = ["src"];

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? walk(path) : [path];
  });
}

/** sha256 over sorted `relpath \0 bytes \0` records (relpath relative to web/, `/`-separated). */
export function sourceStamp(): string {
  const files = [
    ...STAMP_FILES.map((name) => join(ROOT, name)),
    ...STAMP_DIRS.flatMap((dir) => walk(join(ROOT, dir))),
  ]
    .map((path) => ({ path, rel: relative(ROOT, path).split(sep).join("/") }))
    .sort((a, b) => (a.rel < b.rel ? -1 : a.rel > b.rel ? 1 : 0));
  const hash = createHash("sha256");
  for (const { path, rel } of files) {
    hash.update(rel, "utf8");
    hash.update("\0");
    hash.update(readFileSync(path));
    hash.update("\0");
  }
  return hash.digest("hex");
}

/** Stamp every emitted chunk, stylesheet and the HTML shell with the source hash. */
function sourceStampPlugin(): Plugin {
  const stamp = sourceStamp();
  return {
    name: "ariadne-source-stamp",
    apply: "build",
    enforce: "post",
    transformIndexHtml() {
      return [
        {
          tag: "meta",
          attrs: { name: "ariadne-source-sha256", content: stamp },
          injectTo: "head-prepend",
        },
      ];
    },
    generateBundle(_options, bundle) {
      const banner = `/*! ariadne | source-sha256:${stamp} */\n`;
      for (const output of Object.values(bundle)) {
        if (output.type === "chunk") {
          output.code = banner + output.code;
        } else if (output.fileName.endsWith(".css")) {
          output.source = banner + String(output.source);
        }
      }
    },
  };
}

/**
 * The graph module is reached only through `import("./graph/ariadne-graph")`, so React Flow + ELK
 * land in one lazily loaded `ariadne-graph` chunk. Fail the build if that boundary ever erodes:
 * ELK outside the chunk, or the shell entry importing the chunk statically, would let a graph
 * failure take the whole console down.
 */
function graphChunkGuard(): Plugin {
  return {
    name: "ariadne-graph-chunk-guard",
    apply: "build",
    generateBundle(_options, bundle) {
      const chunks = Object.values(bundle).filter((output) => output.type === "chunk");
      const graph = chunks.find((chunk) => chunk.name === GRAPH_CHUNK);
      if (!graph) this.error(`missing lazy chunk "${GRAPH_CHUNK}"`);
      const isElk = (id: string) => /[\\/]elkjs[\\/]lib[\\/]elk\.bundled\.js$/.test(id);
      const isFlow = (id: string) => /[\\/]@xyflow[\\/]react[\\/]/.test(id);
      if (!graph.moduleIds.some(isElk) || !graph.moduleIds.some(isFlow)) {
        this.error(`"${GRAPH_CHUNK}" must contain elk.bundled.js and @xyflow/react`);
      }
      for (const chunk of chunks) {
        if (chunk !== graph && chunk.moduleIds.some((id) => isElk(id) || isFlow(id))) {
          this.error(`${chunk.fileName} must not bundle ELK or React Flow`);
        }
        if (chunk.isEntry && chunk.imports.includes(graph.fileName)) {
          this.error(`${chunk.fileName} statically imports ${graph.fileName}`);
        }
      }
    },
  };
}

const GRAPH_CHUNK = "ariadne-graph";

export default defineConfig({
  root: ROOT,
  base: "/static/",
  plugins: [react(), graphChunkGuard(), sourceStampPlugin()],
  server: {
    // Same-origin API in dev: the FastAPI server (`uv run olympus serve`) answers /api and /healthz.
    proxy: {
      "/api": {
        target: API_TARGET,
        // SSE frames must stream through unbuffered and the long-lived socket must never time out.
        timeout: 0,
        proxyTimeout: 0,
        configure(proxy) {
          proxy.on("proxyRes", (response) => {
            if (String(response.headers["content-type"] || "").startsWith("text/event-stream")) {
              response.headers["cache-control"] = "no-cache";
              response.headers["x-accel-buffering"] = "no";
            }
          });
        },
      },
      "/healthz": { target: API_TARGET },
    },
  },
  build: {
    outDir: OUT_DIR,
    emptyOutDir: true,
    target: "es2022",
    sourcemap: false,
    reportCompressedSize: false,
    chunkSizeWarningLimit: 4096,
    rolldownOptions: {
      output: {
        entryFileNames: "assets/ariadne.js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: (asset) =>
          asset.names?.[0] === "index.css" ? "assets/ariadne.css" : "assets/[name][extname]",
      },
    },
  },
});
