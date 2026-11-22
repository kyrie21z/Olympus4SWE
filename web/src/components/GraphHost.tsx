import { Component, lazy, Suspense, useMemo, useState, type ErrorInfo, type ReactNode } from "react";

import { Notice } from "./ui";
import type { DagGraphProps, WorkflowGraphProps } from "../graph/types";

type GraphModule = typeof import("../graph/ariadne-graph");

let cachedGraphModule: GraphModule | null = null;
let graphModulePromise: Promise<GraphModule> | null = null;
let graphLoadAttempt = 0;

/** Upper bound so a never-firing `load`/`error` event cannot wedge the graph import. */
const STYLESHEET_LOAD_TIMEOUT_MS = 4000;

/**
 * Keep this import unconditional and alone: Vite/Rolldown analyses the literal specifier here and
 * emits `__vite__mapDeps([..., ariadne-graph.css])`. Folding it into a ternary with a
 * `@vite-ignore` sibling made the preload list collapse to `[]` (regression #79).
 */
function importInitialGraphModule(): Promise<GraphModule> {
  return import("../graph/ariadne-graph");
}

function waitForStylesheet(link: HTMLLinkElement): Promise<void> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      link.removeEventListener("load", finish);
      link.removeEventListener("error", finish);
      resolve();
    };
    const timer = setTimeout(finish, STYLESHEET_LOAD_TIMEOUT_MS);
    link.addEventListener("load", finish);
    link.addEventListener("error", finish);
  });
}

/** Resolve once ariadne-graph.css is applied, injecting it when Vite's preload was bypassed. */
async function ensureGraphStylesheet(): Promise<void> {
  if (typeof document === "undefined" || import.meta.env.DEV) return;
  const href = new URL(/* @vite-ignore */ "./ariadne-graph.css", import.meta.url).href;
  const existing = Array.from(document.querySelectorAll('link[rel="stylesheet"]')).find(
    (link) => (link as HTMLLinkElement).href === href,
  ) as HTMLLinkElement | undefined;
  if (existing) {
    if (existing.sheet) return;
    await waitForStylesheet(existing);
    return;
  }
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  const loaded = waitForStylesheet(link);
  document.head.appendChild(link);
  await loaded;
}

function loadGraphModule(): Promise<GraphModule> {
  if (cachedGraphModule) return Promise.resolve(cachedGraphModule);
  if (graphModulePromise) return graphModulePromise;
  const attempt = graphLoadAttempt++;
  const promise: Promise<GraphModule> =
    attempt === 0 || import.meta.env.DEV
      ? importInitialGraphModule()
      : (import(
          /* @vite-ignore */ `${new URL(/* @vite-ignore */ "./ariadne-graph.js", import.meta.url).href}?retry=${attempt}`
        ) as Promise<GraphModule>);
  graphModulePromise = promise
    .then(async (mod) => {
      await ensureGraphStylesheet();
      cachedGraphModule = mod;
      return mod;
    })
    .catch((error) => {
      graphModulePromise = null;
      throw error;
    });
  return graphModulePromise;
}

interface BoundaryProps {
  name: string;
  /** Changing this key (e.g. switching runs) retries a graph that failed to load or render. */
  resetKey: string;
  children: ReactNode;
}

interface BoundaryState {
  error: Error | null;
  resetKey: string;
}

export class GraphErrorBoundary extends Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { error: null, resetKey: this.props.resetKey };

  static getDerivedStateFromError(error: Error): Partial<BoundaryState> {
    return { error };
  }

  static getDerivedStateFromProps(props: BoundaryProps, state: BoundaryState): Partial<BoundaryState> | null {
    return props.resetKey !== state.resetKey ? { error: null, resetKey: props.resetKey } : null;
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.warn(`[ariadne] ${this.props.name} graph unavailable`, error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (error) {
      return (
        <Notice level="warn">
          {`${this.props.name}组件未能加载（ariadne-graph：${error.message || "未知错误"}），其余控制台功能不受影响。`}
        </Notice>
      );
    }
    return this.props.children;
  }
}

export function DagGraphHost(props: DagGraphProps) {
  // Freeze the choice per mount: if `cachedGraphModule` fills in after the lazy chunk resolves, a
  // later re-render would swap <LazyDag> for <Loaded>, remounting the graph and dropping focus and
  // the panned/zoomed viewport on the first interaction.
  const [Loaded] = useState(() => cachedGraphModule?.DagGraphView ?? null);
  const LazyDag = useMemo(
    () =>
      Loaded
        ? null
        : lazy(() => loadGraphModule().then((module) => ({ default: module.DagGraphView }))),
    [Loaded, props.runId],
  );
  return (
    <div className="graph-host" id="dag-graph" data-graph-host="dag">
      <GraphErrorBoundary name="任务依赖图" resetKey={props.runId}>
        {Loaded ? (
          <Loaded {...props} />
        ) : LazyDag ? (
          <Suspense fallback={null}>
            <LazyDag {...props} />
          </Suspense>
        ) : null}
      </GraphErrorBoundary>
    </div>
  );
}

export function WorkflowGraphHost(props: WorkflowGraphProps) {
  const [Loaded] = useState(() => cachedGraphModule?.WorkflowGraphView ?? null);
  const LazyWorkflow = useMemo(
    () =>
      Loaded
        ? null
        : lazy(() => loadGraphModule().then((module) => ({ default: module.WorkflowGraphView }))),
    [Loaded, props.runId],
  );
  return (
    <div className="graph-host" id="workflow-graph" data-graph-host="workflow">
      <GraphErrorBoundary name="编排工作流" resetKey={props.runId}>
        {Loaded ? (
          <Loaded {...props} />
        ) : LazyWorkflow ? (
          <Suspense fallback={null}>
            <LazyWorkflow {...props} />
          </Suspense>
        ) : null}
      </GraphErrorBoundary>
    </div>
  );
}
