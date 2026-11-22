import { Component, lazy, Suspense, useMemo, type ErrorInfo, type ReactNode } from "react";

import { Notice } from "./ui";
import type { DagGraphProps, WorkflowGraphProps } from "../graph/types";

type GraphModule = typeof import("../graph/ariadne-graph");

let cachedGraphModule: GraphModule | null = null;
let graphModulePromise: Promise<GraphModule> | null = null;
let graphLoadAttempt = 0;

function ensureGraphStylesheet(): void {
  if (typeof document === "undefined" || import.meta.env.DEV) return;
  const href = new URL(/* @vite-ignore */ "./ariadne-graph.css", import.meta.url).href;
  const existing = Array.from(document.querySelectorAll('link[rel="stylesheet"]')).some(
    (link) => (link as HTMLLinkElement).href === href,
  );
  if (existing) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  document.head.appendChild(link);
}

function loadGraphModule(): Promise<GraphModule> {
  if (cachedGraphModule) return Promise.resolve(cachedGraphModule);
  if (graphModulePromise) return graphModulePromise;
  const attempt = graphLoadAttempt++;
  const promise: Promise<GraphModule> =
    attempt === 0 || import.meta.env.DEV
      ? import("../graph/ariadne-graph")
      : (import(
          /* @vite-ignore */ `${new URL(/* @vite-ignore */ "./ariadne-graph.js", import.meta.url).href}?retry=${attempt}`
        ) as Promise<GraphModule>);
  graphModulePromise = promise
    .then((mod) => {
      if (attempt > 0) ensureGraphStylesheet();
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
  const Loaded = cachedGraphModule?.DagGraphView ?? null;
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
  const Loaded = cachedGraphModule?.WorkflowGraphView ?? null;
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
