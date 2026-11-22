/**
 * Standalone mount harness for the graph components (browser diagnostics and the Playwright
 * resilience seams: empty, cyclic, retry-loop and high fan-in graphs rendered from raw props).
 *
 * The console itself renders `DagGraphView` / `WorkflowGraphView` declaratively; this harness is
 * registered on `window.AriadneGraph` only once the lazy `ariadne-graph` chunk has loaded.
 */
import { DagGraph } from "./dag/DagGraph";
import { createIsland } from "./island";
import type { AriadneGraphAPI, DagGraphProps, WorkflowGraphProps } from "./types";
import { WorkflowGraph } from "./workflow/WorkflowGraph";

declare global {
  interface Window {
    AriadneGraph?: AriadneGraphAPI;
  }
}

export const AriadneGraph: AriadneGraphAPI = Object.freeze({
  version: "2",
  mountDag: (container: HTMLElement, initialProps: DagGraphProps) =>
    createIsland(container, DagGraph, initialProps),
  mountWorkflow: (container: HTMLElement, initialProps: WorkflowGraphProps) =>
    createIsland(container, WorkflowGraph, initialProps),
});

export function registerGraphHarness(): void {
  window.AriadneGraph = AriadneGraph;
}
