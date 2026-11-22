import { Component, lazy, Suspense, type ErrorInfo, type ReactNode } from "react";

import { Notice } from "./ui";
import type { DagGraphProps, WorkflowGraphProps } from "../graph/types";

const loadGraphModule = () => import("../graph/ariadne-graph");

const LazyDagGraph = lazy(() => loadGraphModule().then((module) => ({ default: module.DagGraphView })));
const LazyWorkflowGraph = lazy(() =>
  loadGraphModule().then((module) => ({ default: module.WorkflowGraphView })),
);

interface BoundaryProps {
  name: string;
  /** Changing this key (e.g. switching runs) retries a graph that failed to render. */
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
  return (
    <div className="graph-host" id="dag-graph" data-graph-host="dag">
      <GraphErrorBoundary name="任务依赖图" resetKey={props.runId}>
        <Suspense fallback={null}>
          <LazyDagGraph {...props} />
        </Suspense>
      </GraphErrorBoundary>
    </div>
  );
}

export function WorkflowGraphHost(props: WorkflowGraphProps) {
  return (
    <div className="graph-host" id="workflow-graph" data-graph-host="workflow">
      <GraphErrorBoundary name="编排工作流" resetKey={props.runId}>
        <Suspense fallback={null}>
          <LazyWorkflowGraph {...props} />
        </Suspense>
      </GraphErrorBoundary>
    </div>
  );
}
