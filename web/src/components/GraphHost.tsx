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

function graphStylesheetUrl(): URL {
  return new URL(/* @vite-ignore */ "./ariadne-graph.css", import.meta.url);
}

function hasLoadedRules(link: HTMLLinkElement): boolean {
  try {
    return Boolean(link.sheet && link.sheet.cssRules.length > 0);
  } catch {
    return false;
  }
}

function findGraphStylesheetLinks(pathname: string): HTMLLinkElement[] {
  return Array.from(document.querySelectorAll('link[rel="stylesheet"]')).filter((node) => {
    const href = (node as HTMLLinkElement).href;
    if (!href) return false;
    try {
      return new URL(href).pathname === pathname;
    } catch {
      return false;
    }
  }) as HTMLLinkElement[];
}

function pruneUnloadedStylesheets(): void {
  if (typeof document === "undefined" || import.meta.env.DEV) return;
  const pathname = graphStylesheetUrl().pathname;
  for (const link of findGraphStylesheetLinks(pathname)) {
    if (!hasLoadedRules(link)) link.remove();
  }
}

function waitForStylesheet(link: HTMLLinkElement): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      clearTimeout(timer);
      link.removeEventListener("load", onLoad);
      link.removeEventListener("error", onError);
    };
    const fail = (reason: string) => {
      if (settled) return;
      settled = true;
      cleanup();
      link.remove();
      reject(new Error(reason));
    };
    const onLoad = () => {
      if (settled) return;
      if (!hasLoadedRules(link)) {
        fail("ariadne-graph.css loaded with no CSS rules");
        return;
      }
      settled = true;
      cleanup();
      resolve();
    };
    const onError = () => fail("ariadne-graph.css failed to load");
    const timer = setTimeout(() => fail("ariadne-graph.css load timed out"), STYLESHEET_LOAD_TIMEOUT_MS);
    link.addEventListener("load", onLoad);
    link.addEventListener("error", onError);
  });
}

/** Resolve once ariadne-graph.css has active rules, injecting or retrying when needed. */
async function ensureGraphStylesheet(attempt: number): Promise<void> {
  if (typeof document === "undefined" || import.meta.env.DEV) return;
  const baseUrl = graphStylesheetUrl();
  const existing = findGraphStylesheetLinks(baseUrl.pathname);
  if (existing.some(hasLoadedRules)) {
    for (const link of existing) {
      if (!hasLoadedRules(link)) link.remove();
    }
    return;
  }
  for (const stale of existing) stale.remove();

  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.crossOrigin = "anonymous";
  link.href = attempt > 0 ? `${baseUrl.href}?retry=${attempt}` : baseUrl.href;
  const loaded = waitForStylesheet(link);
  document.head.appendChild(link);
  await loaded;
}

function loadGraphModule(): Promise<GraphModule> {
  if (cachedGraphModule) return Promise.resolve(cachedGraphModule);
  if (graphModulePromise) return graphModulePromise;
  const attempt = graphLoadAttempt++;
  const importPromise: Promise<GraphModule> =
    attempt === 0 || import.meta.env.DEV
      ? importInitialGraphModule()
      : (import(
          /* @vite-ignore */ `${new URL(/* @vite-ignore */ "./ariadne-graph.js", import.meta.url).href}?retry=${attempt}`
        ) as Promise<GraphModule>);
  // Swallow late rejections if the timeout below rejects this attempt first.
  void importPromise.catch(() => undefined);

  const attemptPromise = new Promise<GraphModule>((resolve, reject) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      if (graphModulePromise === attemptPromise) graphModulePromise = null;
      pruneUnloadedStylesheets();
      reject(new Error("ariadne-graph bundle load timed out"));
    }, STYLESHEET_LOAD_TIMEOUT_MS);

    importPromise
      .then(async (mod) => {
        if (settled) return;
        await ensureGraphStylesheet(attempt);
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        cachedGraphModule = mod;
        resolve(mod);
      })
      .catch((error: unknown) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (graphModulePromise === attemptPromise) graphModulePromise = null;
        pruneUnloadedStylesheets();
        reject(error);
      });
  });

  graphModulePromise = attemptPromise;
  return attemptPromise;
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
